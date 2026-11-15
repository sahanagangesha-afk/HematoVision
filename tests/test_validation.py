import pytest

from hematovision.validation import ValidationError, validate_upload
from tests.conftest import make_image_bytes


def test_valid_png_and_jpeg():
    assert validate_upload("cell.png", make_image_bytes("PNG")).mode == "RGB"
    assert validate_upload("cell.jpg", make_image_bytes("JPEG")).mode == "RGB"


@pytest.mark.parametrize("name", ["", None, "   "])
def test_missing_filename(name):
    with pytest.raises(ValidationError, match="No file"):
        validate_upload(name, make_image_bytes())


def test_bad_extension():
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload("cell.gif", make_image_bytes())
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload("noextension", make_image_bytes())


def test_empty_file():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("cell.png", b"")


def test_corrupted_file():
    with pytest.raises(ValidationError, match="corrupted"):
        validate_upload("cell.png", b"this is definitely not an image")


def test_truncated_image():
    data = make_image_bytes("PNG")
    with pytest.raises(ValidationError):
        validate_upload("cell.png", data[: len(data) // 2])


def test_too_small():
    with pytest.raises(ValidationError, match="too small"):
        validate_upload("cell.png", make_image_bytes(size=(16, 16)))


def test_blank_image_rejected():
    with pytest.raises(ValidationError, match="blank"):
        validate_upload("cell.png", make_image_bytes(noise=False))


def test_too_large_file(monkeypatch):
    monkeypatch.setattr("hematovision.validation.MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ValidationError, match="too large"):
        validate_upload("cell.png", make_image_bytes())
