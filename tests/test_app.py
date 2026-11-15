import io

from tests.conftest import make_image_bytes


def post(client, data, name="cell.png"):
    return client.post("/api/predict", data={"file": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_index_and_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").get_json() == {"status": "ok"}


def test_predict_success(client):
    res = post(client, make_image_bytes())
    assert res.status_code == 200
    body = res.get_json()
    assert body["label"] == "neutrophil"
    assert set(body["probabilities"]) == {"eosinophil", "lymphocyte", "monocyte", "neutrophil"}


def test_no_file_part(client):
    res = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert res.status_code == 400


def test_wrong_extension(client):
    assert post(client, make_image_bytes(), "cell.txt").status_code == 400


def test_corrupt_image(client):
    assert post(client, b"garbage").status_code == 400


def test_blank_image(client):
    assert post(client, make_image_bytes(noise=False)).status_code == 400


def test_huge_upload_returns_413(client):
    assert post(client, b"0" * (8 * 1024 * 1024)).status_code == 413


def test_internal_error_is_hidden(broken_client):
    res = post(broken_client, make_image_bytes())
    assert res.status_code == 500
    assert "boom" not in res.get_data(as_text=True)


def test_missing_model_returns_503(tmp_path, monkeypatch):
    from app import create_app
    monkeypatch.setattr("app.Predictor",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("no model")))
    res = post(create_app().test_client(), make_image_bytes())
    assert res.status_code == 503


def test_unknown_route(client):
    assert client.get("/nope").status_code == 404
