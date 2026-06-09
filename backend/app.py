"""
NIDS (Network Intrusion Detection System) — Flask Backend
Loads XGBoost model + preprocessors and exposes prediction endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# ─── Load Saved Artifacts ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model          = joblib.load(os.path.join(BASE_DIR, "models/xgboost_model.pkl"))
    scaler         = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
    pca            = joblib.load(os.path.join(BASE_DIR, "models/pca.pkl"))
    feature_indices = joblib.load(os.path.join(BASE_DIR, "models/feature_indices.pkl"))
    print("✅ All model artifacts loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = scaler = pca = feature_indices = None

# ─── Feature names (from CICIDS2017 dataset – all 78 cols minus Label) ───────
ALL_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
    "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate", "Subflow Fwd Packets", "Subflow Fwd Bytes",
    "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]


def preprocess(raw_features: dict) -> np.ndarray:
    """
    raw_features: dict of {feature_name: value} for all 78 features.
    Returns PCA-transformed array ready for model.predict().
    """
    # 1. Build a full-feature DataFrame with correct column order
    df = pd.DataFrame([raw_features], columns=ALL_FEATURES)

    # 2. Infinities → NaN → 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # 3. Select the top-30 features chosen during training
    X = df.iloc[:, feature_indices]

    # 4. Scale
    X_scaled = scaler.transform(X)

    # 5. PCA
    X_pca = pca.transform(X_scaled)

    return X_pca


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Quick liveness check."""
    status = "ready" if model is not None else "model_not_loaded"
    return jsonify({"status": status})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON body:
    {
      "features": {
        "Destination Port": 80,
        "Flow Duration": 12345,
        ...all 78 features...
      }
    }
    Returns:
    {
      "prediction": 0 or 1,
      "label": "BENIGN" or "ATTACK",
      "confidence": 0.97
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Place .pkl files in backend/models/"}), 503

    data = request.get_json(force=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key in request body"}), 400

    try:
        X_pca = preprocess(data["features"])
        pred = int(model.predict(X_pca)[0])
        prob = float(model.predict_proba(X_pca)[0][pred])

        return jsonify({
            "prediction": pred,
            "label": "ATTACK" if pred == 1 else "BENIGN",
            "confidence": round(prob * 100, 2),
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Expects JSON body:
    {
      "records": [ {feature_dict}, {feature_dict}, ... ]
    }
    Returns list of predictions.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(force=True)
    if not data or "records" not in data:
        return jsonify({"error": "Missing 'records' key"}), 400

    results = []
    for i, record in enumerate(data["records"]):
        try:
            X_pca = preprocess(record)
            pred  = int(model.predict(X_pca)[0])
            prob  = float(model.predict_proba(X_pca)[0][pred])
            results.append({
                "index": i,
                "prediction": pred,
                "label": "ATTACK" if pred == 1 else "BENIGN",
                "confidence": round(prob * 100, 2),
            })
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    attack_count  = sum(1 for r in results if r.get("label") == "ATTACK")
    benign_count  = sum(1 for r in results if r.get("label") == "BENIGN")

    return jsonify({
        "total": len(results),
        "attacks": attack_count,
        "benign": benign_count,
        "results": results,
    })


@app.route("/features", methods=["GET"])
def features():
    """Returns the list of all expected feature names (useful for the frontend form)."""
    top30 = [ALL_FEATURES[i] for i in feature_indices] if feature_indices is not None else []
    return jsonify({
        "all_features": ALL_FEATURES,
        "top30_features": top30,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
