"""Runtime compatibility for the pinned TF1-era RigL code under TensorFlow 2.15.

The official checkout remains pristine. Set this environment variable before
TensorFlow imports and use the legacy SGD class because upstream logging asks
for ``get_slot_names``/``get_slot``. The original optimizer and mask-update
semantics are unchanged.
"""
import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow.compat.v2 as tf

tf.keras.optimizers.SGD = tf.keras.optimizers.legacy.SGD
