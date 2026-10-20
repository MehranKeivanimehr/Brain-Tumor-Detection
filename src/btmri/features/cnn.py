"""Frozen ImageNet CNN embeddings (global-average-pooled final features).

These are generic pretrained encoders, the standard "deep features + shallow
classifier" baseline used throughout the brain tumor MRI literature (e.g.
Kang, Ullah & Gwak, Sensors 2021, who pair them with ELM and SVM heads).
Weights are the Keras Applications releases (He et al. 2016; Huang et al. 2017).
"""

from __future__ import annotations

import numpy as np

MODELS = {
    "resnet50": ("ResNet50", "resnet"),
    "densenet121": ("DenseNet121", "densenet"),
}


def batch_cnn(images: np.ndarray, name: str = "resnet50", batch: int = 64) -> np.ndarray:
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import keras
    import tensorflow as tf

    cls, mod = MODELS[name]
    model = getattr(keras.applications, cls)(include_top=False, weights="imagenet", pooling="avg")
    preprocess = getattr(keras.applications, mod).preprocess_input
    out = []
    for i in range(0, len(images), batch):
        x = tf.image.resize(images[i : i + batch, :, :, None].astype(np.float32), (224, 224))
        x = tf.repeat(x, 3, axis=-1)
        out.append(model.predict(preprocess(x), verbose=0).astype(np.float32))
    return np.concatenate(out)
