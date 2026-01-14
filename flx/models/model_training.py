from dataclasses import dataclass
import json
from os.path import join, exists

import tqdm
import shutil

import torch
import torchmetrics

from flx.setup.paths import get_best_model_file
from flx.setup.paths import get_newest_model_file
from flx.benchmarks.matchers import CosineSimilarityMatcher
from flx.benchmarks.verification import VerificationBenchmark, VerificationResult
from flx.data.embedding_loader import EmbeddingLoader
from flx.data.dataset import Dataset, ZippedDataLoader
from flx.extractor.extract_embeddings import extract_embeddings
from flx.models.deep_print_arch import DeepPrintTrainingOutput
from flx.setup.config import LEARNING_RATE
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
    validation_equal_error_rate: float

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

    @property
    def best_entry(self) -> TrainingLogEntry:
        return min(self._entries, key=lambda e: e.validation_equal_error_rate)

    def __len__(self) -> int:
        return len(self._entries)

    def add_entry(self, entry: TrainingLogEntry):
        self._entries.append(entry)
        self._save()


def _train(
    model: torch.nn.Module,
    loss_fun: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_set: Dataset,
    sampler: torch.utils.data.Sampler = None,
    use_amp: bool = True,
) -> float:
    """
    Trains the model for one epoch.

    Args:
        use_amp: Use Automatic Mixed Precision (AMP) for 2-3x speedup.
                 Set to False if you encounter numerical instability.

    Returns
        - overall average epoch loss
        - a dict with the epoch loss of individual loss components
    """
    metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes=train_set.num_subjects
    ).to(device=get_device())

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
    loss_fun.reset_recorded_loss()
    for vals in tqdm.tqdm(train_dataloader):
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
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
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
            loss.backward()
            optimizer.step()

        # Record accuracy and loss
        epoch_loss += float(loss) * fp_labels.shape[0]
        if output.combined_logits is not None:
            logits = output.combined_logits
        elif output.minutia_logits is None:
            logits = output.texture_logits
        elif output.texture_logits is None:
            logits = output.minutia_logits
        else:
            logits = output.texture_logits + output.minutia_logits
        metric(logits, fp_labels)

    mean_loss = epoch_loss / len(train_set)
    multiclass_accuracy = float(metric.compute())
    return mean_loss, loss_fun.get_recorded_loss(), multiclass_accuracy


def _validate(
    model: torch.nn.Module,
    validation_set: Dataset,
    benchmark: VerificationBenchmark,
) -> float:
    """
    Validates the model.

    Returns equal error rate
    """
    texture_embeddings, minutia_embeddings = extract_embeddings(model, validation_set)
    embeddings = EmbeddingLoader.combine_if_both_exist(
        texture_embeddings, minutia_embeddings
    )

    matcher = CosineSimilarityMatcher(embeddings)
    result: VerificationResult = benchmark.run(matcher)
    return result.get_equal_error_rate()


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
) -> None:
    """
    Trains model for num_iter and saves results (training log and model parameters)

    Automatically loads model parameters from "model.pyt" file if exists.
    Supports multi-GPU training with DistributedDataParallel when launched with torchrun.

    Args:
        patience: Early stopping patience. If > 0, training stops after this many
                  epochs without validation improvement. Set to 0 to disable.
        use_amp: Use Automatic Mixed Precision for 2-3x speedup and 30-40% less memory.
                 Recommended for large-scale training. Set to False if numerical issues occur.
    """
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
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    loss = loss.to(device=get_device())
    if exists(model_path):
        if is_main_process():
            print(f"Loaded existing model from {model_path}")
        load_model_parameters(model_path, model, loss, optimizer)
    else:
        if is_main_process():
            print(f"No model file found at {model_path}")

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
    # Broadcast start_epoch to all processes in distributed mode
    if is_distributed():
        import torch.distributed as dist
        start_tensor = torch.tensor([start_epoch], device=get_device())
        dist.broadcast(start_tensor, src=0)
        start_epoch = int(start_tensor.item())

    # Early stopping tracking
    epochs_without_improvement = 0
    best_eer = float('inf')

    try:
        for epoch in range(start_epoch, num_epochs + 1):
            # Set epoch for distributed sampler (ensures different shuffling each epoch)
            if sampler is not None:
                sampler.set_epoch(epoch)

            if is_main_process():
                print(f"\n\n --- Starting Epoch {epoch} of {num_epochs} ---")
                print("\nTraining:")

            train_loss, loss_stats, accuracy = _train(
                model, loss, optimizer, training_set, sampler, use_amp
            )

            if is_main_process():
                print(f"Average Loss: {train_loss}")
                print(f"Multiclass accuracy: {accuracy}")

                # Save model (unwrap from DDP if wrapped)
                save_model_parameters(
                    model_path, unwrap_model(model), loss, optimizer
                )

            if validation_fingerprints is None:
                # Use training accuracy as validation accuracy
                validation_eer = accuracy
            else:
                # Validate (only main process needs to do this)
                if is_main_process():
                    print("\nValidation:")
                    validation_eer = _validate(
                        unwrap_model(model), validation_fingerprints, validation_benchmark
                    )
                    print(f"Equal Error Rate: {validation_eer}\n")
                else:
                    validation_eer = 0.0  # Placeholder for non-main processes

            # Log and determine if new model is best model (main process only)
            should_stop = False
            if is_main_process():
                entry = TrainingLogEntry(
                    epoch, train_loss, loss_stats, accuracy, validation_eer
                )
                log.add_entry(entry)
                print(entry)

                if validation_eer <= log.best_entry.validation_equal_error_rate:
                    shutil.copyfile(model_path, best_model_path)

                # Early stopping check
                if validation_fingerprints is not None and patience > 0:
                    if validation_eer < best_eer:
                        best_eer = validation_eer
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        print(f"No improvement for {epochs_without_improvement}/{patience} epochs")

                    if epochs_without_improvement >= patience:
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
