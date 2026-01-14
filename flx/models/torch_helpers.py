import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


CUDA_DEVICE = 0
HIGH_VRAM_GPU = True

# Distributed training state
_distributed_initialized = False


def get_world_size() -> int:
    """Returns the number of GPUs available for training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Returns the global rank of the current process."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Returns the local rank (GPU index on this machine)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Returns True if this is the main process (rank 0)."""
    return get_rank() == 0


def is_distributed() -> bool:
    """Returns True if distributed training is enabled."""
    return dist.is_initialized() and get_world_size() > 1


def setup_distributed(backend: str = "nccl") -> bool:
    """
    Initialize distributed training if multiple GPUs are available.
    Returns True if distributed training was initialized, False otherwise.

    Can be launched with:
        torchrun --nproc_per_node=NUM_GPUS script.py
    Or:
        python -m torch.distributed.launch --nproc_per_node=NUM_GPUS script.py
    """
    global _distributed_initialized

    # Check if already initialized
    if dist.is_initialized():
        _distributed_initialized = True
        return True

    # Check if launched with torchrun/distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        _distributed_initialized = True

        if is_main_process():
            print(f"Distributed training initialized: {world_size} GPUs")

        return True

    # Not launched with distributed, check if we should auto-init
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print(f"Single GPU mode (found {num_gpus} GPU(s))")
        return False

    # Multiple GPUs but not launched with torchrun - inform user
    print(f"Found {num_gpus} GPUs. For multi-GPU training, launch with:")
    print(f"  torchrun --nproc_per_node={num_gpus} your_script.py")
    print("Continuing with single GPU...")
    return False


def cleanup_distributed() -> None:
    """Clean up distributed training."""
    global _distributed_initialized
    if dist.is_initialized():
        dist.destroy_process_group()
        _distributed_initialized = False


def wrap_model_ddp(model: torch.nn.Module) -> torch.nn.Module:
    """Wrap model with DistributedDataParallel if distributed training is enabled."""
    if is_distributed():
        local_rank = get_local_rank()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            print(f"Model wrapped with DistributedDataParallel")
    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap model from DDP if wrapped."""
    if isinstance(model, DDP):
        return model.module
    return model


def get_dataloader_args(train: bool) -> dict:
    """
    Get DataLoader arguments optimized for training scale.

    Batch sizes:
    - Small datasets (<100K samples): batch_size=32
    - Large datasets (>100K samples): batch_size=64-128
    - Adjust based on GPU memory (reduce if OOM errors)

    Workers:
    - Rule of thumb: 2-4 workers per GPU
    - For 4 GPUs: 16 workers, 8 GPUs: 16-32 workers
    """
    # Default batch size for modern GPUs (24GB+ VRAM)
    # Reduce to 32 if you get OOM errors
    batch_size = 64

    if not train:
        batch_size *= 2  # More memory available without gradients

    if not torch.cuda.is_available():
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 4,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }

    if HIGH_VRAM_GPU:  # Use high VRAM GPUs by preloading to pinned memory
        return {
            "batch_size": batch_size,
            "shuffle": train,
            "num_workers": 16,
            "prefetch_factor": 4,  # Increased for better GPU utilization
            "pin_memory": True,
            "persistent_workers": True,  # Keep workers alive between epochs
        }

    return {
        "batch_size": batch_size,
        "shuffle": train,
        "num_workers": 8,  # Increased from 4
        "prefetch_factor": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }


def get_device() -> torch.device:
    """Get the device for the current process (supports distributed training)."""
    if torch.cuda.is_available():
        if is_distributed():
            return torch.device(f"cuda:{get_local_rank()}")
        return torch.device(f"cuda:{CUDA_DEVICE}")
    return torch.device("cpu")


def save_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to save the parameters of model and optimizer in the given path
    """
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "loss_state_dict": loss.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
            },
            full_param_path,
        )
    except KeyboardInterrupt:
        print("\n>>>>>>>>> Model is being saved! Will exit when done <<<<<<<<<<\n")
        save_model_parameters(full_param_path, model, optim)
        time.sleep(10)
        raise KeyboardInterrupt()


def load_model_parameters(
    full_param_path: str,
    model: torch.nn.Module,
    loss: torch.nn.Module,
    optim: torch.optim.Optimizer,
) -> None:
    """
    Tries to load the parameters stored in the given path
    into the given model and optimizer.
    """
    if not os.path.exists(full_param_path):
        raise FileNotFoundError(f"Model file {full_param_path} did not exist.")
    checkpoint = torch.load(full_param_path, map_location=get_device(), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if loss is not None:
        loss.load_state_dict(checkpoint["loss_state_dict"])
    if optim is not None:
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
