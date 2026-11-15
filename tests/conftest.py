import io

import numpy as np
import pytest
from PIL import Image

from app import create_app


def make_image_bytes(fmt="PNG", size=(128, 128), noise=True) -> bytes:
    rng = np.random.default_rng(0)
    arr = (rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
           if noise else np.full((size[1], size[0], 3), 200, dtype=np.uint8))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


class FakePredictor:
    def predict(self, img):
        return {
            "label": "neutrophil", "confidence": 0.91, "uncertain": False, "threshold": 0.6,
            "probabilities": {"eosinophil": 0.02, "lymphocyte": 0.03, "monocyte": 0.04, "neutrophil": 0.91},
        }


class BrokenPredictor:
    def predict(self, img):
        raise RuntimeError("boom")


@pytest.fixture
def client():
    return create_app(FakePredictor()).test_client()


@pytest.fixture
def broken_client():
    return create_app(BrokenPredictor()).test_client()
