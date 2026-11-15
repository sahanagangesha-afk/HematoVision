"""Inference: load the trained model and produce validated predictions."""
import json
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD, DEFAULT_CLASSES, MODEL_PATH
from .preprocessing import preprocess_image


def build_result(probs, class_names, threshold=CONFIDENCE_THRESHOLD) -> dict:
    """Turn a probability vector into a JSON-friendly result.

    Guards against edge cases: wrong vector length, NaN/inf outputs, and
    low-confidence predictions (flagged as ``uncertain``).
    """
    probs = np.asarray(probs, dtype=np.float64).ravel()
    if probs.shape[0] != len(class_names):
        raise ValueError(
            f"Model returned {probs.shape[0]} outputs but {len(class_names)} classes are defined."
        )
    if not np.all(np.isfinite(probs)):
        raise ValueError("Model produced invalid (NaN/inf) output.")

    idx = int(np.argmax(probs))
    confidence = float(probs[idx])
    return {
        "label": class_names[idx],
        "confidence": round(confidence, 4),
        "uncertain": confidence < threshold,
        "threshold": threshold,
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(class_names, probs)
        },
    }


class Predictor:
    """Wraps the Keras model. TensorFlow is imported lazily."""

    def __init__(self, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
        model_path, class_names_path = Path(model_path), Path(class_names_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run `python -m hematovision.train` first."
            )
        import tensorflow as tf  # noqa: WPS433 (lazy import on purpose)

        self.model = tf.keras.models.load_model(model_path)
        if class_names_path.exists():
            self.class_names = json.loads(class_names_path.read_text())
        else:
            self.class_names = DEFAULT_CLASSES

    def predict(self, img) -> dict:
        batch = preprocess_image(img)
        probs = self.model.predict(batch, verbose=0)[0]
        return build_result(probs, self.class_names)
