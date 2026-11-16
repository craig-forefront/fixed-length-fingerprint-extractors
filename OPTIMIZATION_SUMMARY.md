# What the training code does for speed

This page lists the speed-related settings of the training code. An earlier version of it gave speed-up factors,
training times and GPU utilisation figures; none of them was measured, and they were removed. Line numbers of the
code are not given, because they go out of date.

| What | Where | Value |
|------|-------|-------|
| Several GPUs | `scripts/train_500k.py` started with `torchrun --nproc_per_node=N`; `setup_distributed` and `wrap_model_ddp` in `flx/models/torch_helpers.py` | One process per GPU. Not run for this repository's guides (no GPU was available) |
| Images per step and GPU | `--batch_size` of `scripts/train_500k.py`; `set_training_batch_size` in `flx/models/torch_helpers.py` | 64 by default; validation and embedding extraction use twice the value |
| Data-loader workers | `get_dataloader_args` in `flx/models/torch_helpers.py` | 16 per GPU process with `prefetch_factor=4` (8 and 2 if `HIGH_VRAM_GPU` is `False`), 4 without a GPU; `persistent_workers=True`, pinned memory on a GPU. Not an option: edit the function |
| Mixed precision | `use_amp` of `train_model` and of `fit` | On by default when a GPU is used; `use_amp=False` switches it off. `scripts/train_500k.py` has no option for it |
| Early stopping | `--patience` | 3 epochs without a lower validation error rate |
| Checkpoints | `model.pyt` after every epoch, `best_model.pyt` for the best epoch | No checkpoint inside an epoch |

What was measured about speed is the image preparation on the CPU, which takes about 1 second per rolled print
per core; see "What limits the speed" in [LARGE_SCALE_TRAINING_GUIDE.md](LARGE_SCALE_TRAINING_GUIDE.md).

Where to read on:

- [TRAINING_PIPELINE_GUIDE.md](TRAINING_PIPELINE_GUIDE.md): the route from images to a trained model.
- [LARGE_SCALE_TRAINING_GUIDE.md](LARGE_SCALE_TRAINING_GUIDE.md): what matters for large datasets.
- ["Above about one million subjects"](TRAINING_PIPELINE_GUIDE.md#above-about-one-million-subjects): the limits
  of the code as it is.
