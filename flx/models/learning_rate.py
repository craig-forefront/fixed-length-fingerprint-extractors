"""
Learning rate schedule, gradient clipping and the check for a diverged
training.

Terms used here:

- A *step* is one update of the model parameters with one batch of images.
- The *learning rate* is the size of that update.
- *Warm-up* means that the first steps use a smaller learning rate that rises
  in a straight line to the chosen learning rate.
- A training has *diverged* when its loss rises without limit or is no longer
  a finite number. The parameters are then useless.
"""

from dataclasses import dataclass
import math

import torch

from flx.setup import config

SCHEDULES = ("constant", "cosine")

# The warm-up takes at most one step in this many steps of the training
WARMUP_MAX_DIVISOR = 10


class TrainingDivergedError(RuntimeError):
    """The loss is not a finite number or far higher than at the start."""


@dataclass(frozen=True)
class LearningRateConfig:
    """
    Settings of the learning rate. The defaults come from flx/setup/config.py.

    Args:
        learning_rate: Learning rate after the warm-up.
        warmup_steps: Number of steps of the warm-up. In step number i
            (counted from 1) of the warm-up the learning rate is
            learning_rate * i / warmup_steps. 0 switches the warm-up off.
            The warm-up is never longer than a tenth of the whole training
            (see effective_warmup_steps), so that a short training on a small
            dataset still reaches the learning rate.
        schedule: "constant" keeps learning_rate after the warm-up. "cosine"
            lowers it along half a cosine wave so that the last step of the
            training uses learning_rate * final_factor.
        final_factor: See schedule. Only used by "cosine".
        gradient_clip_norm: Upper limit for the L2 norm of the gradient of all
            parameters in one step; a longer gradient is scaled down to this
            length. 0 switches the limit off; None uses GRADIENT_CLIP_NORM
            from the config (None there also means no limit).
        loss_explosion_factor: The training stops with a
            TrainingDivergedError when the loss of a batch is higher than this
            factor times the reference loss (the loss of the first batch of the
            run, or the highest epoch loss in the log of a resumed training if
            that is higher). 0 switches this check off. A loss that is not a
            finite number always stops the training.
    """

    learning_rate: float = None
    warmup_steps: int = None
    schedule: str = None
    final_factor: float = None
    gradient_clip_norm: float | None = None
    loss_explosion_factor: float = None

    def __post_init__(self):
        # The defaults are read here, not in the class body, so that a value
        # changed in flx.setup.config at run time is used.
        defaults = {
            "learning_rate": config.LEARNING_RATE,
            "warmup_steps": config.LEARNING_RATE_WARMUP_STEPS,
            "schedule": config.LEARNING_RATE_SCHEDULE,
            "final_factor": config.LEARNING_RATE_FINAL_FACTOR,
            "loss_explosion_factor": config.LOSS_EXPLOSION_FACTOR,
        }
        for name, value in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, value)
        if self.gradient_clip_norm is None and config.GRADIENT_CLIP_NORM:
            object.__setattr__(self, "gradient_clip_norm", config.GRADIENT_CLIP_NORM)
        if not self.gradient_clip_norm:
            object.__setattr__(self, "gradient_clip_norm", None)

        if not (self.learning_rate > 0 and math.isfinite(self.learning_rate)):
            raise ValueError(f"learning_rate must be above 0, got {self.learning_rate}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must not be negative, got {self.warmup_steps}")
        if self.schedule not in SCHEDULES:
            raise ValueError(
                f"schedule must be one of {SCHEDULES}, got {self.schedule!r}"
            )
        if not 0 <= self.final_factor <= 1:
            raise ValueError(f"final_factor must be in 0 .. 1, got {self.final_factor}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm < 0:
            raise ValueError("gradient_clip_norm must not be negative")
        if self.loss_explosion_factor != 0 and self.loss_explosion_factor <= 1:
            raise ValueError(
                "loss_explosion_factor must be above 1, or 0 to switch the check off"
            )

    def effective_warmup_steps(self, total_steps: int) -> int:
        """
        Number of warm-up steps of a training with total_steps steps:
        warmup_steps, but at most a tenth of total_steps (rounded down).
        """
        return min(self.warmup_steps, total_steps // WARMUP_MAX_DIVISOR)

    def learning_rate_at(self, step: int, total_steps: int) -> float:
        """
        Learning rate of one step.

        The value depends only on the number of the step, so every process of
        a multi-GPU training computes the same value, and a resumed training
        continues exactly where the schedule was, without stored state.

        Args:
            step: Number of steps done before this one in the whole training
                (0 for the first step of epoch 1).
            total_steps: Number of steps of the whole training
                (epochs * steps per epoch).
        """
        warmup_steps = self.effective_warmup_steps(total_steps)
        if step < warmup_steps:
            return self.learning_rate * (step + 1) / warmup_steps
        if self.schedule == "constant":
            return self.learning_rate
        # cosine: progress is 0 in the first step after the warm-up and 1 in
        # the last step of the training
        decay_steps = total_steps - warmup_steps
        if decay_steps <= 1:
            return self.learning_rate
        progress = min((step - warmup_steps) / (decay_steps - 1), 1.0)
        factor = self.final_factor + (1 - self.final_factor) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )
        return self.learning_rate * factor

    def describe(self, total_steps: int | None = None) -> str:
        """
        The settings in words. With total_steps (the number of steps of the
        whole training) the warm-up is given as it will be used.
        """
        text = f"learning rate {self.learning_rate:g}"
        warmup_steps = self.warmup_steps
        if total_steps is not None:
            warmup_steps = self.effective_warmup_steps(total_steps)
        if warmup_steps > 0:
            text += f", warm-up over {warmup_steps} steps"
            if warmup_steps < self.warmup_steps:
                text += (
                    f" (a tenth of the {total_steps} steps of the training "
                    f"instead of {self.warmup_steps})"
                )
        else:
            text += ", no warm-up"
        if self.schedule == "cosine":
            text += (
                f", then cosine decay to {self.learning_rate * self.final_factor:g}"
                " in the last step"
            )
        else:
            text += ", then constant"
        if self.gradient_clip_norm is not None:
            text += f", gradient norm limited to {self.gradient_clip_norm:g}"
        else:
            text += ", gradient norm not limited"
        return text


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def gradient_norm(parameters) -> float:
    """L2 norm of the gradient of all given parameters together."""
    squares = [
        torch.sum(p.grad.detach().float() ** 2)
        for p in parameters
        if p.grad is not None
    ]
    if len(squares) == 0:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()))


class DivergenceGuard:
    """
    Stops a training whose loss is not a finite number or has exploded.

    Call check() with the loss of every batch before the optimizer step, so
    that a bad batch does not change the parameters any further.
    """

    def __init__(self, explosion_factor: float, reference_loss: float | None = None):
        """
        Args:
            explosion_factor: See LearningRateConfig.loss_explosion_factor.
            reference_loss: Loss to compare with. The first checked loss is
                used if it is higher or if this is None.
        """
        self._factor = explosion_factor
        self._reference = None
        if reference_loss is not None and math.isfinite(reference_loss):
            self._reference = reference_loss
        self._first_seen = False

    @property
    def reference_loss(self) -> float | None:
        return self._reference

    def check(self, loss: float, learning_rate: float, where: str) -> None:
        advice = (
            "This usually means that the learning rate is too high. Lower it "
            "(scripts/find_learning_rate.py suggests a value for your data and "
            "batch size), use more warm-up steps, or switch on gradient "
            "clipping. See 'Choosing the learning rate' in "
            "TRAINING_PIPELINE_GUIDE.md. The parameters of this epoch were not "
            "saved: model.pyt, if it exists, is still the end of the last "
            "completed epoch."
        )
        if not math.isfinite(loss):
            raise TrainingDivergedError(
                f"Training stopped: the loss is {loss} (not a finite number) "
                f"at {where} with learning rate {learning_rate:g}. {advice}"
            )
        if not self._first_seen:
            self._first_seen = True
            if self._reference is None or loss > self._reference:
                self._reference = loss
            return
        # A reference of zero or below gives no scale to compare with
        if (
            self._factor > 0
            and self._reference > 0
            and loss > self._factor * self._reference
        ):
            raise TrainingDivergedError(
                f"Training stopped: the loss is {loss:g} at {where} with "
                f"learning rate {learning_rate:g}. That is more than "
                f"{self._factor:g} times the loss at the start "
                f"({self._reference:g}), so the training has diverged. {advice}"
            )
