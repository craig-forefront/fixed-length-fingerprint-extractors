"""
The localization network: learned alignment of the fingerprint image.

The network looks at a small copy of the image and proposes a rotation and a
shift (and, as an option, a change of scale). The image is resampled with
them before it reaches the rest of the model, so that the model sees every
print in a similar pose. Nothing tells the network which pose is right: it is
trained together with the model, by the same loss.

Terms used here:

- *Aligned image*: the image after the resampling.
- *Identity*: no rotation, no shift, scale 1. The aligned image is then the
  input image.
- *Bound*: the largest rotation, shift or scale change the network can
  propose at all. The DeepPrint paper (Engelsma et al. 2019, section 3.2)
  bounds the rotation to +/-60 degrees and the shift to 224 pixels of its
  448 pixel images, "in order to get the localization network to properly
  converge". Without bounds the network of this repository pushed every
  image out of the frame within a few steps.
"""

from dataclasses import dataclass
import math

import torch
import torchvision

from flx.setup import config


@dataclass
class Alignment:
    """
    What the localization network did to one batch of images.

    matrices:     Shape (batch, 2, 3). The affine matrices as given to
                  torch.nn.functional.affine_grid: the point p of the aligned
                  image (coordinates from -1 to 1 in both directions) shows
                  the point  scale * R(rotation) p + 2 * shift  of the input.
    rotation:     Shape (batch,). Rotation in radians.
    shift:        Shape (batch, 2). Shift in x and y as a share of the image
                  side: the centre of the aligned image is the point of the
                  input that lies this far from the input's centre.
    scale:        Shape (batch,). Above 1 the aligned image shows a larger
                  part of the input, so the print gets smaller.
    kept_area:    Shape (batch,). Share of the aligned image that shows a part
                  of the input; the rest lies outside the input and is filled
                  with 0. Not part of the gradient.
    kept_content: Shape (batch,). Sum of the absolute pixel values of the
                  aligned image divided by that of the input. The training
                  pipeline delivers images with background 0 and ridges 1, so
                  this is the share of the print that is still in the frame
                  (the resampling can make it slightly more than 1). Not part
                  of the gradient.
    """

    matrices: torch.Tensor
    rotation: torch.Tensor
    shift: torch.Tensor
    scale: torch.Tensor
    kept_area: torch.Tensor
    kept_content: torch.Tensor

    def detached(self) -> "Alignment":
        return Alignment(
            *(getattr(self, name).detach() for name in self.__dataclass_fields__)
        )


class LocalizationNetwork(torch.torch.nn.Module):
    def __init__(
        self,
        max_rotation_degrees: float | None = None,
        max_shift: float | None = None,
        max_scale_factor: float | None = None,
    ):
        """
        The bounds; None takes the value from flx/setup/config.py
        (LOCALIZATION_MAX_ROTATION_DEGREES, LOCALIZATION_MAX_SHIFT and
        LOCALIZATION_MAX_SCALE_FACTOR, explained there).

        The raw outputs r of the last layer are turned into the rotation
        max_rotation * tanh(r), the shift max_shift * tanh(r) and the scale
        max_scale_factor ** tanh(r). tanh is smooth, close to r for small r
        and never leaves -1 .. 1, so the bounds cannot be exceeded, the
        gradient never vanishes abruptly, and raw outputs of 0 (the
        initialisation) give the identity.

        With max_scale_factor 1.0 the scale is fixed and the last layer has 3
        outputs, as in the DeepPrint paper; above 1.0 it has 4.
        """
        super().__init__()
        if max_rotation_degrees is None:
            max_rotation_degrees = config.LOCALIZATION_MAX_ROTATION_DEGREES
        if max_shift is None:
            max_shift = config.LOCALIZATION_MAX_SHIFT
        if max_scale_factor is None:
            max_scale_factor = config.LOCALIZATION_MAX_SCALE_FACTOR
        if not 0 < max_rotation_degrees <= 180:
            raise ValueError(
                "max_rotation_degrees must be above 0 and at most 180, "
                f"got {max_rotation_degrees}"
            )
        if not 0 < max_shift <= 1:
            raise ValueError(f"max_shift must be above 0 and at most 1, got {max_shift}")
        if not 1 <= max_scale_factor < math.inf:
            raise ValueError(
                f"max_scale_factor must be at least 1, got {max_scale_factor}"
            )
        self.max_rotation: float = math.radians(max_rotation_degrees)
        self.max_shift: float = float(max_shift)
        self.max_scale_factor: float = float(max_scale_factor)
        self.learns_scale: bool = self.max_scale_factor > 1.0

        # The localization net uses a downsampled version of the image for performance
        self.input_size = (128, 128)
        self.resize = torchvision.transforms.Resize(
            size=self.input_size, antialias=True
        )
        # Spatial transformer localization-network
        self.localization = torch.nn.Sequential(
            torch.nn.Conv2d(1, 24, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2, stride=2),
        )

        # Regressor for rotation, shift in x and y, and (as an option) scale
        self.fc_loc = torch.nn.Sequential(
            torch.nn.Linear(8 * 8 * 64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4 if self.learns_scale else 3),
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.zero_()

    def align(self, x: torch.Tensor) -> tuple[torch.Tensor, Alignment]:
        """
        @returns : The aligned images and what was done to them.
        """
        resized_x = self.resize(x)
        xs = self.localization(resized_x)
        xs = xs.view(-1, 8 * 8 * 64)
        # The transform is calculated in 32-bit numbers also when the model
        # runs with mixed precision
        raw = torch.tanh(self.fc_loc(xs).float())
        theta = self.max_rotation * raw[:, 0]
        shift = self.max_shift * raw[:, 1:3]
        if self.learns_scale:
            scale = torch.exp(math.log(self.max_scale_factor) * raw[:, 3])
        else:
            scale = torch.ones_like(theta)
        # Construct rotation and scaling matrix. The grid coordinates run from
        # -1 to 1, so a shift by a share s of the image side is 2 * s there.
        m11 = scale * torch.cos(theta)
        m12 = -scale * torch.sin(theta)
        m13 = 2 * shift[:, 0]
        m21 = scale * torch.sin(theta)
        m22 = scale * torch.cos(theta)
        m23 = 2 * shift[:, 1]

        # One row per sample: stack along dim 1. Concatenating the six vectors
        # end to end mixes the samples of a batch into each other's matrices.
        mat = torch.stack((m11, m12, m13, m21, m22, m23), dim=1).view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(
            mat.to(dtype=x.dtype), x.size(), align_corners=False
        )
        aligned = torch.nn.functional.grid_sample(x, grid, align_corners=False)
        with torch.no_grad():
            inside = (grid.abs() <= 1).all(dim=-1).float()
            kept_area = inside.mean(dim=(1, 2))
            content_in = x.float().abs().flatten(1).sum(dim=1)
            content_out = aligned.float().abs().flatten(1).sum(dim=1)
            # An input without any content cannot lose it
            kept_content = torch.where(
                content_in > 0,
                content_out / content_in.clamp(min=1e-12),
                torch.ones_like(content_in),
            )
        return aligned, Alignment(mat, theta, shift, scale, kept_area, kept_content)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(x)[0]


def main():
    model = LocalizationNetwork()
    print("no syntax errors")


if __name__ == "__main__":
    main()
