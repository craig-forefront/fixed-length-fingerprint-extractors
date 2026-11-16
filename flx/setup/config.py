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

# --- Loss weights ------------------------------------------------------------
# The training loss is a sum of parts. Each part is multiplied by its weight.
# These values are only the defaults of flx.models.deep_print_loss.LossWeights.

# Cross entropy of each classifier (variants with texture and minutiae have
# two): how wrong the model is when it names the subject of a training image.
# It is the mean over the images of a batch.
LOSS_WEIGHT_CROSS_ENTROPY = 1.0

# Center loss of each embedding: the squared distance between the embedding of
# an image and the running average of the embeddings of its subject (the
# "center"). It pulls the embeddings of one finger together. It is the mean
# over the images of a batch.
#
# Until this value was introduced the center loss was the sum over the batch
# with the factor 0.125, so its weight relative to the cross entropy grew with
# the batch size. 8.0 = 0.125 x 64 gives the same loss as before for the batch
# size of train_model (64 images, get_dataloader_args in
# flx/models/torch_helpers.py) and now gives that same balance for every
# batch size. For a batch of B images the former balance is 0.125 x B.
# The value is high: the DeepPrint paper (Engelsma et al. 2019) states 0.00125,
# and at the start of a training on real images the center loss with 8.0 sent
# about 30 times more gradient into the shared first layers than the cross
# entropy (one batch of 8 images, DeepPrint_TexMinu, texture branch). Whether
# a lower value gives better accuracy was not measured. See "Loss weights"
# in TRAINING_PIPELINE_GUIDE.md.
LOSS_WEIGHT_CENTER = 8.0

# Minutia map loss (variants with minutiae only): the squared difference
# between the minutia map that the model draws and the map made from the
# minutiae of the image, as the sum over the cells of the map (6 x 128 x 128 =
# 98,304) and the mean over the images of a batch that have a map.
#
# 0.095 is the value of the DeepPrint paper (Engelsma et al. 2019, equation
# 8). In short runs on real images it made the gradient of the map loss on the
# shared first layers about as long as that of the texture cross entropy (0.4
# to 1.2 times) and 1.5 % to 4.6 % of the whole gradient there. Until this
# value was introduced the map loss was the mean over the cells with the
# factor 0.3, which equals 0.3 / 98,304 = 0.00000305 here; with it the map
# loss made up about 0.001 % of the gradient on these layers, so the minutia
# maps hardly influenced what the shared layers learn. See "Loss weights" in
# TRAINING_PIPELINE_GUIDE.md for the measurement and its limits.
LOSS_WEIGHT_MINUTIA_MAP = 0.095

# Change this when switching models DeepPrint has 299 while ViT has 224
INPUT_SIZE = 299
