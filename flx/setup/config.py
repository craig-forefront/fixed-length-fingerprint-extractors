INTERACTIVE_VIS = False

# --- Learning rate -----------------------------------------------------------
# The learning rate is the size of the change that the optimizer (Adam) makes
# to the model parameters in every training step. The right value depends on
# the data, the model variant and the batch size: find it with
# scripts/find_learning_rate.py (see "Choosing the learning rate" in
# TRAINING_PIPELINE_GUIDE.md). These values are only the defaults of
# flx.models.learning_rate.LearningRateConfig.

# Learning rate after the warm-up. 0.0001 trained DeepPrint_TexMinu and
# DeepPrint_LocTexMinu without divergence in short runs on real images with
# 8 images per step; the former default 0.025 diverged in the second step.
# The value was not tested with large batches or on other data.
LEARNING_RATE = 0.0001

# Number of training steps (batches) over which the learning rate rises in a
# straight line from near zero to LEARNING_RATE. 0 switches the warm-up off.
# The warm-up is never longer than a tenth of all steps of the training.
LEARNING_RATE_WARMUP_STEPS = 500

# What the learning rate does after the warm-up:
#   "constant": it stays at LEARNING_RATE
#   "cosine":   it falls along half a cosine wave to
#               LEARNING_RATE * LEARNING_RATE_FINAL_FACTOR in the last step of
#               the last epoch
LEARNING_RATE_SCHEDULE = "constant"
LEARNING_RATE_FINAL_FACTOR = 0.01

# Upper limit for the length (L2 norm) of the gradient of all parameters in one
# step. A longer gradient is scaled down to this length. None switches the
# limit off.
GRADIENT_CLIP_NORM = None

# The training stops with an error when the loss of a batch is higher than this
# factor times the loss at the start of the training. 0 switches the check off.
# A loss that is not a finite number always stops the training.
LOSS_EXPLOSION_FACTOR = 10.0

# Change this when switching models DeepPrint has 299 while ViT has 224
INPUT_SIZE = 299
