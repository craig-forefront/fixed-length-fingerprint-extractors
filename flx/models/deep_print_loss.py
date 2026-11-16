from dataclasses import dataclass

import torch
from torch import nn

from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.center_loss import CenterLoss
from flx.setup import config


@dataclass(frozen=True)
class LossWeights:
    """
    The factors with which the parts of the training loss are added up. A
    value of None takes the default from flx/setup/config.py, where the parts
    are explained (LOSS_WEIGHT_CROSS_ENTROPY, LOSS_WEIGHT_CENTER and
    LOSS_WEIGHT_MINUTIA_MAP).

    cross_entropy: factor of the cross entropy of each classifier
    center:        factor of the center loss of each embedding
    minutia_map:   factor of the minutia map loss (variants with minutiae only)
    """

    cross_entropy: float | None = None
    center: float | None = None
    minutia_map: float | None = None

    def __post_init__(self):
        defaults = {
            "cross_entropy": config.LOSS_WEIGHT_CROSS_ENTROPY,
            "center": config.LOSS_WEIGHT_CENTER,
            "minutia_map": config.LOSS_WEIGHT_MINUTIA_MAP,
        }
        for name, default in defaults.items():
            value = getattr(self, name)
            if value is None:
                value = default
            value = float(value)
            if not value >= 0 or value == float("inf"):
                raise ValueError(
                    f"The loss weight {name} must be a number that is not "
                    f"negative, found {value}"
                )
            object.__setattr__(self, name, value)


def _compute_minutia_map_loss(
    x: torch.Tensor,
    minutia_maps: torch.Tensor,
    minutia_map_weights: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """
    Squared difference between the predicted maps x and the target maps, as
    the sum over the cells of a map (as in equation 8 of the DeepPrint paper)
    and the weighted mean over the samples.

    The weight of a sample is 1 if it has a target map and 0 if it has none
    (see MinutiaMapLoader.get). The loss is divided by the sum of the weights,
    not by the batch size, so that the map loss counts the same whether or
    not the batch also contains samples without a map. A batch without any
    map has the loss 0.

    The sum is calculated in 32-bit numbers also when the model runs with
    mixed precision, because a sum over about 98,000 cells can be too large
    for 16-bit numbers.

    @returns : The loss and the sum of the weights (the number of samples with a map)
    """
    x = x.float()
    weights = minutia_map_weights.to(dtype=x.dtype)
    mm_squared_diff = (x - minutia_maps.to(dtype=x.dtype)) ** 2
    mm_sse = mm_squared_diff.reshape(weights.shape[0], -1).sum(dim=1)
    weight_sum = weights.sum()
    # Without any map both sums are 0; the lower limit avoids 0 / 0
    loss = (mm_sse * weights).sum() / torch.clamp(weight_sum, min=1e-12)
    return loss, float(weight_sum.detach().item())


class _DeepPrint_Embedding_Loss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_embedding_dims: int,
        weights: LossWeights | None = None,
    ):
        """
        @param weights : Factors of the cross entropy and the center loss. None uses the defaults.
        """
        super().__init__()
        self.weights: LossWeights = weights if weights is not None else LossWeights()
        self.crossent_loss_fun = nn.CrossEntropyLoss()
        self.center_loss_fun = CenterLoss(num_classes, num_embedding_dims)
        self.crossent_loss_sum: float = 0
        self.center_loss_sum: float = 0
        self.n_datapoints: int = 0

    def reset_recorded_loss(self) -> None:
        self.crossent_loss_sum = 0
        self.center_loss_sum = 0
        self.n_datapoints = 0

    def get_recorded_loss(self) -> dict:
        """
        Returns the mean loss per sample since the last reset, each part
        multiplied by its weight. The keys end in "_sum" for compatibility
        with existing training logs; the values are means per sample.
        """
        if self.n_datapoints == 0:
            return {"crossent_loss_sum": 0, "center_loss_sum": 0}
        return {
            "crossent_loss_sum": self.crossent_loss_sum / self.n_datapoints,
            "center_loss_sum": self.center_loss_sum / self.n_datapoints,
        }

    def forward(
        self, embeddings: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        crossent_loss = self.crossent_loss_fun(logits, labels)
        center_loss = self.center_loss_fun(embeddings, labels)
        # Both parts are means over the batch, so they count once per sample
        self.crossent_loss_sum += (
            crossent_loss.detach().item()
            * self.weights.cross_entropy
            * labels.shape[0]
        )
        self.center_loss_sum += (
            center_loss.detach().item() * self.weights.center * labels.shape[0]
        )
        self.n_datapoints += labels.shape[0]
        return (
            self.weights.cross_entropy * crossent_loss
            + self.weights.center * center_loss
        )


class DeepPrintLoss_Tex(nn.Module):
    """
    Loss of the variants without minutiae: cross entropy and center loss of
    the texture embedding.
    """

    def __init__(
        self,
        num_classes: int,
        texture_embedding_dims: int,
        weights: LossWeights | None = None,
    ):
        """
        @param weights : Factors of the parts of the loss. None uses the defaults from flx/setup/config.py.
        """
        super().__init__()
        self.weights: LossWeights = weights if weights is not None else LossWeights()
        self.texture_loss_fun: _DeepPrint_Embedding_Loss = _DeepPrint_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=texture_embedding_dims,
            weights=self.weights,
        )

    def forward(
        self,
        output: DeepPrintTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepPrintTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        if minutia_maps.shape[1] != 0:
            raise RuntimeWarning(
                "DeepPrint_TextureLoss received non-empty minutia_maps!"
            )
        texture_loss = self.texture_loss_fun(
            output.texture_embeddings, output.texture_logits, labels
        )
        return texture_loss

    def get_recorded_loss(self) -> dict:
        return self.texture_loss_fun.get_recorded_loss()

    def reset_recorded_loss(self) -> None:
        self.texture_loss_fun.reset_recorded_loss()


class DeepPrintLoss_Minu(nn.Module):
    def __init__(
        self,
        num_classes: int,
        minutia_embedding_dims: int,
        weights: LossWeights | None = None,
    ):
        """
        @param weights : Factors of the parts of the loss. None uses the defaults from flx/setup/config.py.
        """
        super().__init__()
        self.weights: LossWeights = weights if weights is not None else LossWeights()
        self.minu_loss_fun: _DeepPrint_Embedding_Loss = _DeepPrint_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=minutia_embedding_dims,
            weights=self.weights,
        )
        self.minu_map_loss_sum: float = 0
        self.n_minu_maps: float = 0

    def forward(
        self,
        output: DeepPrintTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepPrintTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        minutia_loss = self.minu_loss_fun(
            output.minutia_embeddings, output.minutia_logits, labels
        )
        mm_loss, n_with_map = _compute_minutia_map_loss(
            output.minutia_maps, minutia_maps, minutia_map_weights
        )
        # The map loss is the mean over the samples that have a map
        self.minu_map_loss_sum += (
            self.weights.minutia_map * mm_loss.detach().item() * n_with_map
        )
        self.n_minu_maps += n_with_map
        return minutia_loss + self.weights.minutia_map * mm_loss

    def get_recorded_loss(self) -> dict:
        return {
            "minutia_loss": self.minu_loss_fun.get_recorded_loss(),
            # Mean over the samples that had a map (0 if none had one)
            "minutia_map_loss": self.minu_map_loss_sum / self.n_minu_maps
            if self.n_minu_maps > 0
            else 0,
        }

    def reset_recorded_loss(self) -> None:
        self.minu_loss_fun.reset_recorded_loss()
        self.minu_map_loss_sum = 0
        self.n_minu_maps = 0


class DeepPrintLoss_TexMinu(nn.Module):
    """
    Loss of the variants with texture and minutiae: cross entropy and center
    loss of both embeddings, and the minutia map loss.
    """

    def __init__(
        self,
        num_classes: int,
        texture_embedding_dims: int,
        minutia_embedding_dims: int,
        weights: LossWeights | None = None,
    ):
        """
        @param weights : Factors of the parts of the loss. None uses the defaults from flx/setup/config.py.
        """
        super().__init__()
        self.weights: LossWeights = weights if weights is not None else LossWeights()
        self.minu_loss_fun: _DeepPrint_Embedding_Loss = _DeepPrint_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=minutia_embedding_dims,
            weights=self.weights,
        )
        self.texture_loss_fun: _DeepPrint_Embedding_Loss = _DeepPrint_Embedding_Loss(
            num_classes=num_classes,
            num_embedding_dims=texture_embedding_dims,
            weights=self.weights,
        )
        self.minu_map_loss_sum: float = 0
        self.n_minu_maps: float = 0

    def forward(
        self,
        output: DeepPrintTrainingOutput,
        labels: torch.Tensor,
        minutia_maps: torch.Tensor,
        minutia_map_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        @param output : Output of the model for the given fingerprint images; Object of type DeepPrintTrainingOutput
        @param labels : Ground truth subject ids for fingerprints images
        @param minutia_maps : Ground truth minutia maps for the fingerprint images
        """
        minutia_loss = self.minu_loss_fun(
            output.minutia_embeddings, output.minutia_logits, labels
        )
        texture_loss = self.texture_loss_fun(
            output.texture_embeddings, output.texture_logits, labels
        )
        mm_loss, n_with_map = _compute_minutia_map_loss(
            output.minutia_maps, minutia_maps, minutia_map_weights
        )
        # The map loss is the mean over the samples that have a map
        self.minu_map_loss_sum += (
            self.weights.minutia_map * mm_loss.detach().item() * n_with_map
        )
        self.n_minu_maps += n_with_map
        return texture_loss + minutia_loss + self.weights.minutia_map * mm_loss

    def get_recorded_loss(self) -> dict:
        return {
            "texture_loss": self.texture_loss_fun.get_recorded_loss(),
            "minutia_loss": self.minu_loss_fun.get_recorded_loss(),
            # Mean over the samples that had a map (0 if none had one)
            "minutia_map_loss": self.minu_map_loss_sum / self.n_minu_maps
            if self.n_minu_maps > 0
            else 0,
        }

    def reset_recorded_loss(self) -> None:
        self.texture_loss_fun.reset_recorded_loss()
        self.minu_loss_fun.reset_recorded_loss()
        self.minu_map_loss_sum = 0
        self.n_minu_maps = 0


def main():
    loss = DeepPrintLoss_Tex(100, 96)
    loss = DeepPrintLoss_TexMinu(100, 96, 96)
    print("No syntax errors")


if __name__ == "__main__":
    main()
