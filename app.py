"""Flask web app for HematoVision.   Run:  python app.py"""
import logging

from flask import Flask, jsonify, render_template, request

from hematovision.config import MAX_UPLOAD_BYTES
from hematovision.predict import Predictor
from hematovision.validation import ValidationError, validate_upload

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hematovision")


def create_app(predictor=None) -> Flask:
    """App factory. Pass a predictor (or a fake one in tests) to skip loading TensorFlow."""
    app = Flask(__name__)
    # Slightly above the limit so our own validator produces the friendly message
    # for files just over 5 MB; Flask rejects anything far bigger with 413.
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024
    state = {"predictor": predictor}

    def get_predictor():
        if state["predictor"] is None:
            state["predictor"] = Predictor()  # may raise FileNotFoundError
        return state["predictor"]

    def run_prediction():
        """Shared logic for the form and JSON endpoints. Returns (payload, status)."""
        file = request.files.get("file")
        if file is None:
            return {"error": "No file part in the request (field name must be 'file')."}, 400
        try:
            img = validate_upload(file.filename, file.read())
        except ValidationError as exc:
            return {"error": str(exc)}, 400

        try:
            result = get_predictor().predict(img)
        except FileNotFoundError as exc:
            logger.error("Model unavailable: %s", exc)
            return {"error": "Model is not available yet. Train it first (see README)."}, 503
        except Exception:  # noqa: BLE001 - never leak stack traces to the client
            logger.exception("Prediction failed")
            return {"error": "Prediction failed. Please try a different image."}, 500
        return result, 200

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/api/predict")
    def api_predict():
        payload, status = run_prediction()
        return jsonify(payload), status

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="File is too large (max 5 MB)."), 413

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found."), 404

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=True)
