from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
SAMPLE_RATE = 16000
FEATURE_COUNT = 13


def extract_features(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {path}")
    return np.mean(
        librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=FEATURE_COUNT),
        axis=1,
    )


def make_placeholder_model():
    rng = np.random.default_rng(42)
    male_features = rng.normal(loc=-2.0, scale=1.0, size=(200, FEATURE_COUNT))
    female_features = rng.normal(loc=2.0, scale=1.0, size=(200, FEATURE_COUNT))
    features = np.vstack((male_features, female_features))
    labels = np.array(["male"] * len(male_features) + ["female"] * len(female_features))
    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(transformed, labels)
    classifier.placeholder_model = True
    return classifier, scaler


def main():
    features = []
    labels = []
    for label in ("male", "female"):
        files = sorted((DATASET_DIR / label).glob("*.wav"))
        for path in files:
            features.append(extract_features(path))
            labels.append(label)

    if not features:
        print(
            "No dataset WAV files found. Creating a synthetic placeholder model; "
            "train again with real labeled audio before using predictions."
        )
        classifier, scaler = make_placeholder_model()
    else:
        if set(labels) != {"male", "female"}:
            raise ValueError(
                "Training requires at least one valid WAV file in both "
                "dataset/male/ and dataset/female/."
            )
        scaler = StandardScaler()
        transformed = scaler.fit_transform(np.asarray(features))
        classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        classifier.fit(transformed, np.asarray(labels))
        classifier.placeholder_model = False
        print(f"Trained on {len(labels)} labeled WAV files.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"Saved model and scaler to {MODEL_DIR}.")


if __name__ == "__main__":
    main()
