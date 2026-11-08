import logging
import os
import uuid
from pathlib import Path

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_PATH = BASE_DIR / "model" / "model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
ALLOWED_EXTENSION = ".wav"

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
CORS(app)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
IS_PLACEHOLDER_MODEL = getattr(model, "placeholder_model", False)
if IS_PLACEHOLDER_MODEL:
    app.logger.warning("Running with a synthetic placeholder model; predictions are not meaningful.")


def extract_features(path):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    if audio.size == 0:
        raise ValueError("The audio file is empty.")
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
    return np.mean(mfcc, axis=1).reshape(1, -1)


@app.get("/")
def home():
    return render_template("index.html", is_placeholder_model=IS_PLACEHOLDER_MODEL)


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify(error="Please choose a WAV file."), 400
    if Path(secure_filename(uploaded_file.filename)).suffix.lower() != ALLOWED_EXTENSION:
        return jsonify(error="Only .wav audio files are supported."), 400

    upload_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.wav"
    try:
        uploaded_file.save(upload_path)
        features = extract_features(str(upload_path))
        probabilities = model.predict_proba(scaler.transform(features))[0]
        predicted_label = str(model.classes_[int(np.argmax(probabilities))]).lower()
        if predicted_label not in {"male", "female"}:
            app.logger.error("Model returned unsupported class: %s", predicted_label)
            return jsonify(error="The model returned an unsupported prediction."), 500
        return jsonify(
            gender=predicted_label,
            confidence=round(float(np.max(probabilities)), 4),
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Unable to process uploaded audio.")
        return (
            jsonify(error="Unable to analyze this audio file. Please upload a valid WAV file."),
            400,
        )
    finally:
        upload_path.unlink(missing_ok=True)


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return jsonify(error="The upload exceeds the 16 MB limit."), 413


@app.errorhandler(404)
def handle_not_found(_error):
    return jsonify(error="The requested endpoint was not found."), 404


@app.errorhandler(405)
def handle_method_not_allowed(_error):
    return jsonify(error="This method is not allowed for the requested endpoint."), 405


@app.errorhandler(500)
def handle_internal_error(_error):
    return jsonify(error="An unexpected server error occurred."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
