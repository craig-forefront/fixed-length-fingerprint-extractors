"""
A cheap check of the learned alignment during training.

The localization network of the Loc* variants decides which part of every
image the rest of the model sees. If it moves the prints out of the frame,
the model trains on empty images. The loss does not show this: in an earlier
run it fell from 12.8 to 8.5 while every image was out of the frame. The
check here uses two numbers that the localization network reports for every
image (see flx.models.localization_network.Alignment):

- kept_area:    share of the aligned image that shows a part of the input
- kept_content: share of the pixel content of the input that is still in
                the aligned image

An aligned image counts as *mostly empty* when one of the two is below
ALIGNMENT_EMPTY_BELOW (see flx/setup/config.py). The cost is two sums over
the batch, which is nothing next to the model itself.
"""

import math

import torch
import tqdm

from flx.models.learning_rate import TrainingDivergedError
from flx.models.localization_network import Alignment
from flx.setup import config


class AlignmentCollapsedError(TrainingDivergedError):
    """The learned alignment moved most prints of a batch out of the frame."""


class AlignmentMonitor:
    """
    Call check() with the alignment of every batch before the optimizer step.
    It warns once per epoch when the share of mostly empty aligned images in
    a batch is above warn_share, and raises AlignmentCollapsedError when it
    is above stop_share. summary() gives the numbers of the epoch for the
    training log; reset() starts a new epoch.
    """

    def __init__(
        self,
        empty_below: float | None = None,
        warn_share: float | None = None,
        stop_share: float | None = None,
    ):
        """
        None takes the value from flx/setup/config.py (ALIGNMENT_EMPTY_BELOW,
        ALIGNMENT_WARN_SHARE, ALIGNMENT_STOP_SHARE). A stop_share of 0
        switches the stop off, a warn_share of 0 the warning.
        """
        self.empty_below = (
            config.ALIGNMENT_EMPTY_BELOW if empty_below is None else empty_below
        )
        self.warn_share = (
            config.ALIGNMENT_WARN_SHARE if warn_share is None else warn_share
        )
        self.stop_share = (
            config.ALIGNMENT_STOP_SHARE if stop_share is None else stop_share
        )
        if not 0 < self.empty_below <= 1:
            raise ValueError("empty_below must be above 0 and at most 1")
        for name in ("warn_share", "stop_share"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 0 and below 1")
        self.reset()

    def reset(self) -> None:
        self._num_samples = 0
        self._num_empty = 0
        self._num_batches_warned = 0
        self._sums = {
            "rotation": 0.0, "shift": 0.0, "kept_area": 0.0, "kept_content": 0.0,
        }
        self._max_rotation = 0.0
        self._max_shift = 0.0
        self._min_scale = math.inf
        self._max_scale = 0.0
        self._min_kept_area = math.inf
        self._min_kept_content = math.inf

    def mostly_empty(self, alignment: Alignment) -> torch.Tensor:
        """One boolean per image of the batch."""
        return (alignment.kept_content < self.empty_below) | (
            alignment.kept_area < self.empty_below
        )

    def check(self, alignment: Alignment | None, where: str) -> None:
        """
        Args:
            alignment: Alignment of the batch; None (a variant without
                localization network) does nothing.
            where: Position in the training, for the messages.
        """
        if alignment is None:
            return
        alignment = alignment.detached()
        empty = self.mostly_empty(alignment)
        n = int(empty.shape[0])
        n_empty = int(empty.sum())
        rotation = alignment.rotation.abs().float()
        # The larger of the shifts in x and in y
        shift = alignment.shift.abs().float().amax(dim=1)

        self._num_samples += n
        self._num_empty += n_empty
        self._sums["rotation"] += float(rotation.sum())
        self._sums["shift"] += float(shift.sum())
        self._sums["kept_area"] += float(alignment.kept_area.sum())
        self._sums["kept_content"] += float(alignment.kept_content.sum())
        self._max_rotation = max(self._max_rotation, float(rotation.max()))
        self._max_shift = max(self._max_shift, float(shift.max()))
        self._min_scale = min(self._min_scale, float(alignment.scale.min()))
        self._max_scale = max(self._max_scale, float(alignment.scale.max()))
        self._min_kept_area = min(self._min_kept_area, float(alignment.kept_area.min()))
        self._min_kept_content = min(
            self._min_kept_content, float(alignment.kept_content.min())
        )

        share = n_empty / n
        description = (
            f"{n_empty} of the {n} aligned images of the batch at {where} are "
            f"mostly empty: less than {self.empty_below:.0%} of the frame "
            "shows the input image, or less than that share of the print is "
            "still in the frame (lowest share of the frame in this batch "
            f"{float(alignment.kept_area.min()):.0%}, lowest share of the print "
            f"{float(alignment.kept_content.min()):.0%}; largest rotation "
            f"{math.degrees(float(rotation.max())):.1f} degrees, largest shift "
            f"{float(shift.max()):.2f} of the image side)."
        )
        if self.stop_share > 0 and share > self.stop_share:
            raise AlignmentCollapsedError(
                f"Training stopped: {description} The localization network "
                "(learned alignment) moves the prints out of the frame, so the "
                "model would train on empty images. Lower the learning rate of "
                "the localization network (LOCALIZATION_LEARNING_RATE_FACTOR), "
                "tighten its bounds (LOCALIZATION_MAX_SHIFT, "
                "LOCALIZATION_MAX_ROTATION_DEGREES), or train a variant "
                "without localization network. See 'Learned alignment' in "
                "docs/training-guide.md. The parameters of this epoch "
                "were not saved."
            )
        if self.warn_share > 0 and share > self.warn_share:
            self._num_batches_warned += 1
            if self._num_batches_warned == 1:
                tqdm.tqdm.write(
                    f"Warning: {description} The training goes on; it stops "
                    f"when more than {self.stop_share:.0%} of a batch are "
                    "mostly empty. Further batches of this epoch are counted "
                    "in the training log and not reported one by one."
                )

    def summary(self) -> dict | None:
        """
        The numbers of the batches since the last reset, or None if there
        was no alignment. Rotations are in degrees, shifts in shares of the
        image side; both are absolute values.
        """
        if self._num_samples == 0:
            return None
        n = self._num_samples
        return {
            "num_images": n,
            "mean_rotation_degrees": math.degrees(self._sums["rotation"] / n),
            "max_rotation_degrees": math.degrees(self._max_rotation),
            "mean_shift": self._sums["shift"] / n,
            "max_shift": self._max_shift,
            "min_scale": self._min_scale,
            "max_scale": self._max_scale,
            "mean_kept_area": self._sums["kept_area"] / n,
            "min_kept_area": self._min_kept_area,
            "mean_kept_content": self._sums["kept_content"] / n,
            "min_kept_content": self._min_kept_content,
            "share_mostly_empty": self._num_empty / n,
            "num_batches_warned": self._num_batches_warned,
        }
