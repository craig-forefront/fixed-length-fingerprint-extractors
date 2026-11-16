from dataclasses import dataclass
import json
import math
import os
from os.path import join, exists

import tqdm
import shutil

import torch

from flx.setup.paths import get_best_model_file
from flx.setup.paths import get_newest_model_file
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.benchmarks.verification import VerificationBenchmark, VerificationResult
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.dataset import Dataset, ZippedDataLoader
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.models.alignment_check import AlignmentMonitor
from flx.models.learning_rate import (
    DivergenceGuard,
    LearningRateConfig,
    localization_learning_rate,
    parameter_groups,
    set_learning_rate,
    set_localization_factor,
)
from flx.models.torch_helpers import (
    get_device,
    load_model_parameters,
    save_model_parameters,
    get_dataloader_args,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    unwrap_model,
    is_main_process,
    is_distributed,
    get_world_size,
)


@dataclass
class TrainingLogEntry:
    epoch: int
    training_loss: float
    loss_statistics: float
    training_accuracy: float
    # Equal error rate on the validation set: the error rate at the decision
    # threshold where false matches and false non-matches are equally frequent.
    # None if the epoch was trained without a validation set.
    validation_equal_error_rate: float | None = None
    # Learning rate of the last step of the epoch. None in logs that were
    # written before this field existed.
    learning_rate: float | None = None
    # What the localization network (learned alignment) did in the epoch, see
    # AlignmentMonitor.summary. None for variants without localization
    # network and in logs that were written before this field existed.
    alignment: dict | None = None

    def __str__(self):
        s = "TrainingLogEntry(\n"
        for k, v in self.__dict__.items():
            s += f"    {k}={v},\n"
        return s + "}"


class TrainingLog:
    def __init__(self, path: str, reset: bool = False):
        self._path: str = path
        self._entries: list[TrainingLogEntry] = []
        if not reset and exists(path):
            self._load()
        else:
            self._save()

    def _save(self):
        with open(self._path, "w") as file:
            obj = {"entries": [e.__dict__ for e in self._entries]}
            json.dump(obj, file)

    def _load(self):
        with open(self._path, "r") as file:
            obj = json.load(file)
            self._entries = [TrainingLogEntry(**dct) for dct in obj["entries"]]
        # Logs written before the learning rate was recorded stored the
        # training accuracy in the error rate field when there was no
        # validation set. Such a value is not an error rate. Newer logs store
        # None in that case, so in them equal values are a real result.
        if all(e.learning_rate is None for e in self._entries):
            for e in self._entries:
                if e.validation_equal_error_rate == e.training_accuracy:
                    e.validation_equal_error_rate = None

    @property
    def selection_metric(self) -> str:
        """
        Name of the field of TrainingLogEntry that decides which epoch is the
        best one. Lower is better for both possible values.

        - "validation_equal_error_rate" if at least one epoch was validated.
          Epochs without a validation result are then never the best.
        - "training_loss" otherwise. The training loss says how well the model
          fits the training data, not how well it compares unseen fingers, so
          it only protects against keeping an epoch after the training diverged.
        """
        if any(e.validation_equal_error_rate is not None for e in self._entries):
            return "validation_equal_error_rate"
        return "training_loss"

    @property
    def best_entry(self) -> TrainingLogEntry | None:
        """
        The entry with the lowest value of `selection_metric`; the earliest one
        if several are equal. Values that are missing or not a number never win
        against a number. None if the log is empty.
        """
        if len(self._entries) == 0:
            return None
        metric = self.selection_metric

        def key(e: TrainingLogEntry) -> float:
            value = getattr(e, metric)
            if value is None or math.isnan(value):
                return math.inf
            return value

        return min(self._entries, key=key)

    @property
    def epochs_since_best(self) -> int:
        """Number of entries that were added after the best entry."""
        if len(self._entries) == 0:
            return 0
        return len(self._entries) - 1 - self._entries.index(self.best_entry)

    @property
    def highest_training_loss(self) -> float | None:
        """Highest finite training loss of all entries; None if there is none."""
        losses = [
            e.training_loss
            for e in self._entries
            if e.training_loss is not None and math.isfinite(e.training_loss)
        ]
        return max(losses) if losses else None

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def last_entry(self) -> TrainingLogEntry | None:
        return self._entries[-1] if self._entries else None

    def add_entry(self, entry: TrainingLogEntry):
        self._entries.append(entry)
        self._save()

    def set_validation_result(
        self, entry: TrainingLogEntry, equal_error_rate: float
    ) -> None:
        """Stores the validation result of an entry that is already in the log."""
        entry.validation_equal_error_rate = equal_error_rate
        self._save()


def _train(
    model: torch.nn.Module,
    loss_fun: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set: Dataset,
    sampler: torch.utils.data.Sampler = None,
    use_amp: bool = True,
    learning_rate: LearningRateConfig | None = None,
    epoch: int = 1,
    num_epochs: int = 1,
    guard: DivergenceGuard | None = None,
    alignment_monitor: AlignmentMonitor | None = None,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        use_amp: Use Automatic Mixed Precision (AMP) for 2-3x speedup.
                 Set to False if you encounter numerical instability.
        learning_rate: Learning rate, warm-up, schedule and gradient clipping.
                 None leaves the learning rate of the optimizer as it is and
                 does not clip.
        epoch: Number of this epoch, counted from 1, and
        num_epochs: number of epochs of the whole training. Together with the
                 number of batches they give the position in the schedule.
        guard: Stops the training when the loss is not finite or has exploded.
        alignment_monitor: Warns or stops the training when the localization
                 network moves the prints out of the frame, and collects what
                 it did in the epoch.

    Raises
        TrainingDivergedError if the guard finds a diverged training.
        AlignmentCollapsedError (a TrainingDivergedError) if the monitor finds
        that most aligned images of a batch are mostly empty.

    Returns
        - overall average epoch loss: the mean over the samples that this
          process trained on (with a sampler these are not the whole set)
        - a dict with the epoch loss of individual loss components
        - the share of these samples whose subject was named correctly
    """
    # Samples whose subject was named correctly. Counted directly: the
    # MulticlassAccuracy of torchmetrics that was used here allocates
    # (number of classes)^2 x 8 bytes in every step, which is 16 GB for
    # 45,000 training subjects and failed for 100,000.
    num_correct = 0

    # Initialize mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None

    dataloader_args = get_dataloader_args(train=True)
    if sampler is not None:
        # When using a sampler, shuffle must be False (sampler handles shuffling)
        dataloader_args["shuffle"] = False
        dataloader_args["sampler"] = sampler
    train_dataloader = torch.utils.data.DataLoader(
        train_set, **dataloader_args
    )
    model.train()  # Outputs minutia maps and logits
    epoch_loss = 0
    num_samples = 0
    loss_fun.reset_recorded_loss()
    # Position in the schedule. Every epoch has the same number of steps, so
    # the steps of the epochs before this one are known without stored state.
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs
    step = steps_per_epoch * (epoch - 1)
    clip_norm = learning_rate.gradient_clip_norm if learning_rate else None
    progress = tqdm.tqdm(train_dataloader)
    for vals in progress:
        if learning_rate is not None:
            set_learning_rate(optimizer, learning_rate.learning_rate_at(step, total_steps))
        current_lr = optimizer.param_groups[0]["lr"]
        step += 1
        fp_imgs, minu_map_tpl, fp_labels = vals
        minu_maps, minu_map_weights = minu_map_tpl
        fp_imgs = fp_imgs.to(device=get_device())
        fp_labels = fp_labels.to(device=get_device())
        minu_maps = minu_maps.to(device=get_device())
        minu_map_weights = minu_map_weights.to(device=get_device())

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        if scaler is not None:
            # Mixed precision training (2-3x faster)
            with torch.cuda.amp.autocast():
                output: DeepPrintTrainingOutput = model(fp_imgs)
                loss = loss_fun.forward(
                    output=output,
                    labels=fp_labels,
                    minutia_maps=minu_maps,
                    minutia_map_weights=minu_map_weights,
                )
            loss_value = loss.detach().item()
            if guard is not None:
                guard.check(loss_value, current_lr, f"epoch {epoch}, step {step}")
            if alignment_monitor is not None:
                alignment_monitor.check(
                    output.alignment, f"epoch {epoch}, step {step}"
                )
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            if clip_norm is not None:
                # The limit applies to the true gradient, not the scaled one
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision training
            output: DeepPrintTrainingOutput = model(fp_imgs)
            loss = loss_fun.forward(
                output=output,
                labels=fp_labels,
                minutia_maps=minu_maps,
                minutia_map_weights=minu_map_weights,
            )
            loss_value = loss.detach().item()
            if guard is not None:
                guard.check(loss_value, current_lr, f"epoch {epoch}, step {step}")
            if alignment_monitor is not None:
                alignment_monitor.check(
                    output.alignment, f"epoch {epoch}, step {step}"
                )
            loss.backward()
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        # Record accuracy and loss
        epoch_loss += loss_value * fp_labels.shape[0]
        num_samples += fp_labels.shape[0]
        if output.combined_logits is not None:
            logits = output.combined_logits
        elif output.minutia_logits is None:
            logits = output.texture_logits
        elif output.texture_logits is None:
            logits = output.minutia_logits
        else:
            logits = output.texture_logits + output.minutia_logits
        num_correct += int((logits.detach().argmax(dim=1) == fp_labels).sum())
        progress.set_postfix(loss=f"{loss_value:.4g}", lr=f"{current_lr:.3g}")

    mean_loss = epoch_loss / num_samples if num_samples > 0 else 0.0
    multiclass_accuracy = num_correct / num_samples if num_samples > 0 else 0.0
    return mean_loss, loss_fun.get_recorded_loss(), multiclass_accuracy


def _validate(
    model: torch.nn.Module,
    validation_set: Dataset,
    benchmark: VerificationBenchmark,
) -> float:
    """
    Validates the model.

    Returns the equal error rate: the error rate at the decision threshold
    where false matches and false non-matches are equally frequent (0 is best).
    """
    texture_embeddings, minutia_embeddings = extract_embeddings(model, validation_set)
    embeddings = EmbeddingLoader.combine_if_both_exist(
        texture_embeddings, minutia_embeddings
    )

    matcher = CosineSimilarityMatcher(embeddings)
    result: VerificationResult = benchmark.run(matcher)
    return result.get_equal_error_rate()


def _record_validation(
    log: TrainingLog,
    entry: TrainingLogEntry,
    validation_eer: float | None,
    model_path: str,
    best_model_path: str,
) -> None:
    """
    Stores the validation result of the newest epoch (None without a
    validation set) and copies model.pyt, which holds that epoch, to
    best_model.pyt if nothing before it was as good.
    """
    if validation_eer is not None:
        log.set_validation_result(entry, validation_eer)
    if log.best_entry is entry:
        # Under a temporary name first: a crash during the copy must not
        # leave half a best_model.pyt
        temporary = f"{best_model_path}.tmp{os.getpid()}"
        shutil.copyfile(model_path, temporary)
        os.replace(temporary, best_model_path)
        print(
            f"Saved epoch {entry.epoch} as best model "
            f"(lowest {log.selection_metric} so far)"
        )


def train_model(
    fingerprints: Dataset,
    minutia_maps: Dataset,
    labels: Dataset,
    validation_fingerprints: Dataset,
    validation_benchmark: VerificationBenchmark,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    num_epochs: int,
    out_dir: str,
    patience: int = 0,
    use_amp: bool = True,
    learning_rate: LearningRateConfig | None = None,
) -> None:
    """
    Trains model for num_iter and saves results (training log and model parameters)

    Automatically loads model parameters from "model.pyt" file if exists.
    Supports multi-GPU training with DistributedDataParallel when launched with torchrun.

    After every epoch the model is saved as "model.pyt". It is also copied to
    "best_model.pyt" if the epoch is the best one so far:

    - With a validation set (validation_fingerprints and validation_benchmark),
      the best epoch is the one with the lowest equal error rate on it.
    - Without a validation set (both None), no error rate is measured and the
      log stores None for it. The best epoch is then the one with the lowest
      training loss. This does not measure how well the model compares fingers
      it has not seen; hold out validation subjects whenever possible.

    Args:
        patience: Early stopping patience. If > 0, training stops after this many
                  epochs without validation improvement. Set to 0 to disable.
                  Has no effect without a validation set. The epochs are counted
                  from the training log, so they carry over when a training is
                  resumed.
        use_amp: Use Automatic Mixed Precision for 2-3x speedup and 30-40% less memory.
                 Recommended for large-scale training. Set to False if numerical issues occur.
        learning_rate: Learning rate, warm-up, schedule, gradient clipping and
                 the limit at which a rising loss stops the training. None uses
                 the defaults from flx/setup/config.py. The schedule depends
                 only on the number of the step, so it continues correctly when
                 a training is resumed with the same num_epochs, training set
                 and batch size, and all processes of a multi-GPU training use
                 the same learning rate.

    Raises:
        TrainingDivergedError: The loss of a batch is not a finite number, or
                 it is far higher than at the start. model.pyt is then still
                 the end of the last completed epoch.
        AlignmentCollapsedError: (a TrainingDivergedError) The localization
                 network of a Loc* variant moved most prints of a batch out
                 of the frame, see flx/models/alignment_check.py.
    """
    if (validation_fingerprints is None) != (validation_benchmark is None):
        raise ValueError(
            "validation_fingerprints and validation_benchmark must be given "
            "together or both be None"
        )
    if learning_rate is None:
        learning_rate = LearningRateConfig()
    # Initialize distributed training if available
    setup_distributed()

    if is_main_process():
        print(f"Using device {get_device()}")
        if is_distributed():
            print(f"Distributed training with {get_world_size()} GPUs")

    # Create output directory and log file
    best_model_path = get_best_model_file(out_dir)
    model_path = get_newest_model_file(out_dir)
    log = TrainingLog(join(out_dir, "log.json")) if is_main_process() else None

    model = model.to(device=get_device())
    # The localization network of the Loc* variants gets its own, smaller
    # learning rate (a second parameter group)
    optimizer = torch.optim.Adam(
        params=parameter_groups(model, learning_rate.localization_factor),
        lr=learning_rate.learning_rate,
    )
    set_learning_rate(optimizer, learning_rate.learning_rate)
    loss = loss.to(device=get_device())
    if exists(model_path):
        if is_main_process():
            print(f"Loaded existing model from {model_path}")
        load_model_parameters(model_path, model, loss, optimizer)
        # The stored optimizer state holds the localization factor of the run
        # that wrote the file. The factor of this run is the one that counts.
        stored_factor = set_localization_factor(
            optimizer, learning_rate.localization_factor
        )
        set_learning_rate(optimizer, learning_rate.learning_rate)
        if (
            is_main_process()
            and stored_factor is not None
            and stored_factor != learning_rate.localization_factor
        ):
            print(
                "The stored training used the localization learning rate "
                f"factor {stored_factor:g}; this run continues with "
                f"{learning_rate.localization_factor:g}."
            )
    else:
        if is_main_process():
            print(f"No model file found at {model_path}")

    # model.pyt and its log entry are written before the validation of an
    # epoch. If the run was stopped during that validation, the entry has no
    # error rate yet: measure it now, on the model that was just loaded.
    if (
        is_main_process()
        and validation_fingerprints is not None
        and validation_benchmark is not None
        and exists(model_path)
        and log.last_entry is not None
        and log.last_entry.validation_equal_error_rate is None
    ):
        print(
            f"Epoch {log.last_entry.epoch} was trained but not validated "
            "(the run was stopped before its validation finished). "
            "Validating it now."
        )
        _record_validation(
            log,
            log.last_entry,
            _validate(model, validation_fingerprints, validation_benchmark),
            model_path,
            best_model_path,
        )

    # Wrap model with DDP for distributed training
    model = wrap_model_ddp(model)

    training_set = Dataset.zip(fingerprints, minutia_maps, labels)

    # Create distributed sampler if in distributed mode
    sampler = None
    if is_distributed():
        sampler = torch.utils.data.distributed.DistributedSampler(
            training_set, shuffle=True
        )

    # Determine starting epoch (only main process has log)
    start_epoch = (len(log) + 1) if is_main_process() else 1
    if is_main_process():
        if (
            validation_fingerprints is not None
            and patience > 0
            and log.epochs_since_best >= patience
        ):
            # The stored training already ended by early stopping (or its
            # last validation, measured above, ends it now)
            print(
                f"The training in {out_dir} already went {log.epochs_since_best} "
                f"epochs without a lower error rate (patience {patience}): "
                "nothing more is trained. best_model.pyt is epoch "
                f"{log.best_entry.epoch}."
            )
            start_epoch = num_epochs + 1
        elif len(log) > 0 and start_epoch > num_epochs:
            print(
                f"The training in {out_dir} already has {len(log)} of "
                f"{num_epochs} epochs: nothing more is trained. Raise the "
                "number of epochs to continue it, or use another output "
                "directory to train again with other settings."
            )
        elif len(log) > 0:
            print(f"Continuing the training in {out_dir} with epoch {start_epoch}.")
    # In a resumed training the loss may already be far below the loss at the
    # start, so the check for an exploded loss compares with the highest epoch
    # loss in the log (-1 if there is none)
    logged_loss = log.highest_training_loss if is_main_process() else None
    logged_loss = -1.0 if logged_loss is None else logged_loss
    # Broadcast start_epoch to all processes in distributed mode
    if is_distributed():
        import torch.distributed as dist
        start_tensor = torch.tensor([start_epoch], device=get_device())
        dist.broadcast(start_tensor, src=0)
        start_epoch = int(start_tensor.item())
        loss_tensor = torch.tensor(
            [logged_loss], dtype=torch.float64, device=get_device()
        )
        dist.broadcast(loss_tensor, src=0)
        logged_loss = float(loss_tensor.item())
    guard = DivergenceGuard(
        learning_rate.loss_explosion_factor,
        reference_loss=logged_loss if logged_loss >= 0 else None,
    )
    # Does nothing for variants without localization network
    alignment_monitor = AlignmentMonitor()
    if is_main_process():
        # Number of steps of the whole training in this process, as _train
        # counts them: the batches of one epoch times the epochs
        samples_per_epoch = len(sampler) if sampler is not None else len(training_set)
        steps_per_epoch = math.ceil(
            samples_per_epoch / get_dataloader_args(train=True)["batch_size"]
        )
        print(
            "Optimizer: Adam, "
            + learning_rate.describe(total_steps=steps_per_epoch * num_epochs)
        )
        if localization_learning_rate(optimizer) is not None:
            print(
                "Localization network: learning rate times "
                f"{learning_rate.localization_factor:g} in every step "
                f"({learning_rate.learning_rate * learning_rate.localization_factor:g} "
                "after the warm-up)"
            )

    if validation_fingerprints is None and is_main_process():
        print(
            "No validation set: best_model.pyt will be the epoch with the "
            "lowest training loss, and early stopping is off."
        )

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)

            if is_main_process():
                print(f"\n\n --- Starting Epoch {epoch} of {num_epochs} ---")
                print("\nTraining:")

            alignment_monitor.reset()
            train_loss, loss_stats, accuracy = _train(
                model,
                loss,
                optimizer,
                training_set,
                sampler,
                use_amp,
                learning_rate=learning_rate,
                epoch=epoch,
                num_epochs=num_epochs,
                guard=guard,
                alignment_monitor=alignment_monitor,
            )
            # With several GPUs these are the images of the main process only
            alignment_stats = alignment_monitor.summary()

            if is_main_process():
                print(f"Average Loss: {train_loss}")
                print(f"Multiclass accuracy: {accuracy}")
                if alignment_stats is not None:
                    print(
                        "Learned alignment: mean rotation "
                        f"{alignment_stats['mean_rotation_degrees']:.2f} degrees "
                        f"(largest {alignment_stats['max_rotation_degrees']:.2f}), "
                        f"mean shift {alignment_stats['mean_shift']:.3f} of the "
                        f"image side (largest {alignment_stats['max_shift']:.3f}), "
                        "share of the print kept in the frame "
                        f"{alignment_stats['mean_kept_content']:.1%} on average "
                        f"(lowest {alignment_stats['min_kept_content']:.1%}), "
                        f"{alignment_stats['share_mostly_empty']:.1%} of the "
                        "aligned images mostly empty"
                    )

                # Save model (unwrap from DDP if wrapped)
                save_model_parameters(
                    model_path, unwrap_model(model), loss, optimizer
                )

                # The log entry is written together with model.pyt, before
                # the validation, so that the two always describe the same
                # epoch. A run that is stopped during a long validation then
                # resumes with the next epoch, not with this one again.
                entry = TrainingLogEntry(
                    epoch,
                    train_loss,
                    loss_stats,
                    accuracy,
                    None,
                    optimizer.param_groups[0]["lr"],
                    alignment_stats,
                )
                log.add_entry(entry)

            # Validate (only the main process needs to do this), log the
            # result and determine if the new model is the best model
            should_stop = False
            if is_main_process():
                validation_eer = None
                if validation_fingerprints is not None:
                    print("\nValidation:")
                    validation_eer = _validate(
                        unwrap_model(model),
                        validation_fingerprints,
                        validation_benchmark,
                    )
                    print(f"Equal Error Rate: {validation_eer}\n")
                _record_validation(
                    log, entry, validation_eer, model_path, best_model_path
                )
                print(entry)

                # Early stopping check
                if validation_fingerprints is not None and patience > 0:
                    epochs_without_improvement = log.epochs_since_best
                    if epochs_without_improvement > 0:
                        print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

                    if epochs_without_improvement >= patience:
                        best_eer = log.best_entry.validation_equal_error_rate
                        print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                        print(f"Best validation EER: {best_eer:.4f}")
                        should_stop = True

            # Broadcast early stopping decision to all processes
            if is_distributed():
                import torch.distributed as dist
                stop_tensor = torch.tensor([1 if should_stop else 0], device=get_device())
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break
    finally:
        # Clean up distributed training
        cleanup_distributed()
