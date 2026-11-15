import numpy as np
import pytest
from PIL import Image

from hematovision.config import IMG_SIZE
from hematovision.predict import build_result
from hematovision.preprocessing import preprocess_image

CLASSES = ["eosinophil", "lymphocyte", "monocyte", "neutrophil"]


def test_preprocess_shape_and_dtype():
    out = preprocess_image(Image.new("RGB", (300, 200), (10, 20, 30)))
    assert out.shape == (1, *IMG_SIZE, 3)
    assert out.dtype == np.float32


def test_preprocess_handles_grayscale_and_rgba():
    assert preprocess_image(Image.new("L", (64, 64))).shape[-1] == 3
    assert preprocess_image(Image.new("RGBA", (64, 64))).shape[-1] == 3


def test_build_result_confident():
    r = build_result([0.05, 0.05, 0.1, 0.8], CLASSES)
    assert r["label"] == "neutrophil" and not r["uncertain"]
    assert r["confidence"] == 0.8


def test_build_result_low_confidence_flagged():
    r = build_result([0.3, 0.3, 0.2, 0.2], CLASSES)
    assert r["uncertain"] is True


def test_build_result_rejects_nan():
    with pytest.raises(ValueError, match="invalid"):
        build_result([np.nan, 0.5, 0.2, 0.3], CLASSES)


def test_build_result_rejects_wrong_length():
    with pytest.raises(ValueError, match="outputs"):
        build_result([0.5, 0.5], CLASSES)
