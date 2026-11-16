"""
Learning rate range test.

The test trains a fresh model for a few hundred steps and multiplies the
learning rate by a constant factor after every step, from a value that is far
too low to a value that is far too high. The loss per step shows where the
model starts to learn and where the training diverges. The test only gives a
starting point: it is short, so a rate that survives it can still be too high
for a long training.
"""

import csv
from dataclasses import dataclass
import math

import numpy as np
import torch
import tqdm

from flx.data.dataset import Dataset
from flx.models.learning_rate import (
    gradient_norm,
    parameter_groups,
    set_learning_rate,
)
from flx.models.torch_helpers import get_device
from flx.setup import config

# Columns of the CSV file
CSV_COLUMNS = ("step", "learning_rate", "loss", "smoothed_loss", "gradient_norm")


@dataclass
class RangeTestResult:
    learning_rates: list[float]
    losses: list[float]
    gradient_norms: list[float]
    # Why the test ended: "all steps done", "loss not finite" or "loss exploded"
    stop_reason: str


@dataclass
class RangeTestSuggestion:
    # Rate at which the smoothed loss fell fastest; None if it never fell
    steepest_learning_rate: float | None
    # Rate at which the smoothed loss was lowest
    lowest_loss_learning_rate: float
    # Rate to try first in a real training; None if the loss never fell
    suggested_learning_rate: float | None
    explanation: str


def range_test_learning_rates(
    min_lr: float, max_lr: float, num_steps: int
) -> list[float]:
    """num_steps rates from min_lr to max_lr, each a constant factor above the last."""
    if not 0 < min_lr < max_lr:
        raise ValueError("min_lr must be above 0 and below max_lr")
    if num_steps < 2:
        raise ValueError("num_steps must be at least 2")
    factor = (max_lr / min_lr) ** (1 / (num_steps - 1))
    return [min_lr * factor**i for i in range(num_steps)]


def smooth_losses(losses: list[float], smoothing: float = 0.9) -> list[float]:
    """
    Exponential moving average of the losses: each value is `smoothing` times
    the previous average plus (1 - smoothing) times the new loss, corrected so
    that the first values are not pulled towards zero. A single batch loss is
    noisy; the average shows the trend. Values that are not finite are kept as
    they are and do not enter the average.
    """
    if not 0 <= smoothing < 1:
        raise ValueError("smoothing must be at least 0 and below 1")
    out = []
    average = 0.0
    n = 0
    for loss in losses:
        if not math.isfinite(loss):
            out.append(loss)
            continue
        n += 1
        average = smoothing * average + (1 - smoothing) * loss
        out.append(average / (1 - smoothing**n))
    return out


def suggest_learning_rate(
    learning_rates: list[float],
    losses: list[float],
    smoothing: float = 0.9,
    safety_divisor: float = 10.0,
    skip_first: int = 5,
    max_factor_below_lowest: float = 100.0,
) -> RangeTestSuggestion:
    """
    Reads a starting learning rate off the result of a range test.

    The smoothed loss is followed up to its lowest point; everything after it
    is the divergence. Before that point, the rate at which the smoothed loss
    falls fastest per factor of the learning rate is the "steepest" rate (the
    fall is measured across a window of a tenth of the steps). The
    suggestion is the steepest rate divided by safety_divisor: at the steepest
    rate the test is already close to the rates that diverge, and a real
    training runs far longer than the test.

    A part of the loss can fall from the first step at any learning rate. The
    minutia map loss does: an untrained model draws a map that is far from
    empty, and tiny changes correct that. The steepest fall is then found at
    the very low rates at the start of the test and says nothing about the
    learning rate. Therefore, if the steepest rate is more than
    max_factor_below_lowest times below the rate with the lowest loss, the
    suggestion is the rate with the lowest loss divided by safety_divisor.

    Args:
        skip_first: Number of steps at the start that are not considered,
            because the average of very few losses is still noisy.
        max_factor_below_lowest: See above.
    """
    if len(learning_rates) != len(losses):
        raise ValueError("learning_rates and losses must have the same length")
    smoothed = np.array(smooth_losses(losses, smoothing), dtype=float)
    finite = np.isfinite(smoothed)
    if finite.sum() < skip_first + 3:
        raise ValueError(
            f"The range test has only {int(finite.sum())} steps with a finite "
            "loss. Start it at a lower learning rate (--min_lr)."
        )
    usable = np.where(finite, smoothed, np.inf)
    i_min = int(np.argmin(usable))
    lowest_lr = float(learning_rates[i_min])

    log_lr = np.log10(np.array(learning_rates, dtype=float))
    steepest_lr = None
    if i_min - skip_first >= 2:
        # Fall of the smoothed loss per factor of ten of the learning rate,
        # measured across a window of steps around each step. The difference
        # between two neighbouring steps would mostly be noise.
        # The window ends at the lowest point at the latest.
        half_window = max(1, min(len(learning_rates) // 20, (i_min - skip_first) // 2))
        index = np.arange(skip_first + half_window, i_min - half_window + 1)
        before = index - half_window
        after = index + half_window
        slope = (usable[after] - usable[before]) / (log_lr[after] - log_lr[before])
        i_steep = int(index[np.argmin(slope)])
        if slope.min() < 0:
            steepest_lr = float(learning_rates[i_steep])

    if steepest_lr is None:
        return RangeTestSuggestion(
            None,
            lowest_lr,
            None,
            "The loss did not fall during the test, so no rate can be "
            "suggested. Start the test at a lower learning rate (--min_lr), "
            "or use more steps.",
        )
    if steepest_lr * max_factor_below_lowest < lowest_lr:
        suggested = lowest_lr / safety_divisor
        return RangeTestSuggestion(
            steepest_lr,
            lowest_lr,
            suggested,
            f"The smoothed loss fell fastest at learning rate {steepest_lr:.3g}, "
            f"more than {max_factor_below_lowest:g} times below the rate "
            f"where it was lowest ({lowest_lr:.3g}). A fall that early comes "
            "from a part of the loss that falls at any rate (for example the "
            "minutia map loss in the first steps), not from the learning "
            f"rate. Suggested starting value: {suggested:.3g} (the rate with "
            f"the lowest loss divided by {safety_divisor:g}). Check it with a "
            "real training run and watch the loss in the first thousand steps.",
        )
    suggested = steepest_lr / safety_divisor
    return RangeTestSuggestion(
        steepest_lr,
        lowest_lr,
        suggested,
        f"The smoothed loss fell fastest at learning rate {steepest_lr:.3g} and "
        f"was lowest at {lowest_lr:.3g}; above that it rose. Suggested "
        f"starting value: {suggested:.3g} (the steepest rate divided by "
        f"{safety_divisor:g}). Check it with a real training run and watch "
        "the loss in the first thousand steps.",
    )


def run_range_test(
    model: torch.nn.Module,
    loss_fun: torch.nn.Module,
    train_set: Dataset,
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_steps: int = 300,
    batch_size: int = 64,
    gradient_clip_norm: float | None = None,
    stop_factor: float = 4.0,
    smoothing: float = 0.9,
    num_workers: int = 0,
    use_amp: bool = True,
    seed: int = 0,
    localization_factor: float | None = None,
) -> RangeTestResult:
    """
    Trains `model` on batches of `train_set` (as zipped by train_model: image,
    (minutia map, weight), label) while raising the learning rate from min_lr
    to max_lr. Uses the optimizer of train_model (Adam). The model and the
    loss are changed and are of no use afterwards.

    The test ends early when the loss is not a finite number or when the
    smoothed loss is above stop_factor times its lowest value so far.

    Args:
        gradient_clip_norm: Same meaning as in LearningRateConfig. Use the
            value of the planned training, because it changes the result.
        use_amp: Use mixed precision on a GPU, as train_model does by default.
        localization_factor: Factor of the learning rate of the localization
            network (Loc* variants), as in LearningRateConfig. The rates of
            the result are those of the other parameters. None uses
            LOCALIZATION_LEARNING_RATE_FACTOR from the config.
    """
    if localization_factor is None:
        localization_factor = config.LOCALIZATION_LEARNING_RATE_FACTOR
    rates = range_test_learning_rates(min_lr, max_lr, num_steps)
    device = get_device()
    model = model.to(device)
    loss_fun = loss_fun.to(device)
    optimizer = torch.optim.Adam(
        params=parameter_groups(model, localization_factor), lr=min_lr
    )
    scaler = (
        torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=len(train_set) > batch_size,
        num_workers=num_workers,
        generator=generator,
    )
    model.train()
    loss_fun.reset_recorded_loss()

    result = RangeTestResult([], [], [], "all steps done")
    lowest_smoothed = math.inf
    progress = tqdm.tqdm(total=num_steps)
    batches = iter(dataloader)
    for rate in rates:
        try:
            vals = next(batches)
        except StopIteration:
            # The subset is smaller than the test: go through it again
            batches = iter(dataloader)
            vals = next(batches)
        fp_imgs, (minu_maps, minu_map_weights), fp_labels = vals
        fp_imgs = fp_imgs.to(device)
        fp_labels = fp_labels.to(device)
        minu_maps = minu_maps.to(device)
        minu_map_weights = minu_map_weights.to(device)

        set_learning_rate(optimizer, rate)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            output = model(fp_imgs)
            loss = loss_fun.forward(
                output=output,
                labels=fp_labels,
                minutia_maps=minu_maps,
                minutia_map_weights=minu_map_weights,
            )
        loss_value = loss.detach().item()
        result.learning_rates.append(rate)
        result.losses.append(loss_value)
        if not math.isfinite(loss_value):
            result.gradient_norms.append(math.nan)
            result.stop_reason = "loss not finite"
            break

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        result.gradient_norms.append(gradient_norm(model.parameters()))
        if gradient_clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        smoothed = smooth_losses(result.losses, smoothing)[-1]
        lowest_smoothed = min(lowest_smoothed, smoothed)
        progress.update(1)
        progress.set_postfix(lr=f"{rate:.3g}", loss=f"{loss_value:.4g}")
        if smoothed > stop_factor * lowest_smoothed:
            result.stop_reason = "loss exploded"
            break
    progress.close()
    return result


def save_range_test_csv(
    path: str, result: RangeTestResult, smoothing: float = 0.9
) -> None:
    smoothed = smooth_losses(result.losses, smoothing)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for i, row in enumerate(
            zip(result.learning_rates, result.losses, smoothed, result.gradient_norms)
        ):
            writer.writerow([i + 1] + [repr(float(v)) for v in row])


def save_range_test_plot(
    path: str,
    result: RangeTestResult,
    suggestion: RangeTestSuggestion,
    smoothing: float = 0.9,
    title: str = "Learning rate range test",
) -> None:
    """
    Plot of the loss against the learning rate (logarithmic axis). The loss
    axis is cut off a little above the loss of the first steps, so that the
    explosion at the right end does not flatten the rest of the curve.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = np.array(result.learning_rates)
    losses = np.array(result.losses)
    smoothed = np.array(smooth_losses(result.losses, smoothing))
    finite = np.isfinite(smoothed)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(
        rates[finite],
        losses[finite],
        color="#9aa0a6",
        linewidth=0.8,
        label="loss of each batch",
    )
    ax.plot(
        rates[finite],
        smoothed[finite],
        color="#1f77b4",
        linewidth=2,
        label=f"smoothed loss (moving average, {smoothing:g})",
    )
    if suggestion.steepest_learning_rate is not None:
        ax.axvline(
            suggestion.steepest_learning_rate,
            color="#d95f02",
            linestyle="--",
            label=f"steepest fall: {suggestion.steepest_learning_rate:.2g}",
        )
        ax.axvline(
            suggestion.suggested_learning_rate,
            color="#1b9e77",
            linestyle="-",
            label=f"suggested start: {suggestion.suggested_learning_rate:.2g}",
        )
    ax.set_xscale("log")
    if finite.any():
        start = float(np.nanmax(smoothed[finite][: max(5, len(rates) // 10)]))
        low = float(np.nanmin(smoothed[finite]))
        margin = 0.15 * max(start - low, 1e-9)
        ax.set_ylim(low - margin, start + 3 * margin)
    ax.set_xlabel("learning rate (rises with every step)")
    ax.set_ylabel("training loss")
    ax.set_title(f"{title}\nended after {len(rates)} steps: {result.stop_reason}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
