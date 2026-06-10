"""
SmartGrid Security System — Deep Learning Backend
Detects anomalies in power grid data using LSTM and Autoencoder models.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os
import traceback
from datetime import datetime
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

# ─── Load Saved Models ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    # Deep Learning Models
    lstm_model = keras.models.load_model(
        os.path.join(BASE_DIR, "models/lstm_anomaly_detector.h5"),
        compile=False
    )
    autoencoder = keras.models.load_model(
        os.path.join(BASE_DIR, "models/autoencoder.h5"),
        compile=False
    )
    
    # Preprocessors
    scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
    
    print("✅ All SmartGrid models loaded successfully.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    lstm_model = autoencoder = scaler = None

# ─── SmartGrid Features ───────────────────────────────────────────────────
GRID_FEATURES = [
    "Voltage", "Current", "Frequency", "Power Factor",
    "Active Power", "Reactive Power", "Apparent Power",
    "Harmonic Distortion", "Phase Angle", "Load Demand",
    "Generation Output", "Net Flow", "Temperature",
    "Timestamp"
]

RISK_LEVELS = {
    "safe": 0,
    "warning": 1,
    "critical": 2
}

# ─── Health Check ────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "SmartGrid Security System",
        "version": "1.0.0",
        "status": "operational",
        "models_loaded": lstm_model is not None and autoencoder is not None
    })

# ─── Anomaly Detection Endpoint ───────────────────────────────────────────
@app.route("/detect-anomaly", methods=["POST"])
def detect_anomaly():
    """
    Detects anomalies in grid data using LSTM and Autoencoder
    
    Expected JSON:
    {
        "voltage": 230.5,
        "current": 45.2,
        "frequency": 50.0,
        "power_factor": 0.98,
        "active_power": 10.5,
        "reactive_power": 2.1,
        "apparent_power": 10.7,
        "harmonic_distortion": 3.2,
        "phase_angle": 11.3,
        "load_demand": 85.0,
        "generation_output": 90.0,
        "net_flow": 5.0,
        "temperature": 35.5
    }
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Extract features in order
        features = np.array([[
            data.get("voltage", 230),
            data.get("current", 45),
            data.get("frequency", 50),
            data.get("power_factor", 0.95),
            data.get("active_power", 10),
            data.get("reactive_power", 2),
            data.get("apparent_power", 10),
            data.get("harmonic_distortion", 3),
            data.get("phase_angle", 10),
            data.get("load_demand", 80),
            data.get("generation_output", 85),
            data.get("net_flow", 5),
            data.get("temperature", 35)
        ]])
        
        # Normalize features
        features_scaled = scaler.transform(features)
        
        # LSTM Prediction
        lstm_prediction = lstm_model.predict(features_scaled.reshape(1, 1, -1), verbose=0)
        lstm_anomaly_score = float(lstm_prediction[0][0])
        
        # Autoencoder Reconstruction Error
        reconstruction = autoencoder.predict(features_scaled, verbose=0)
        reconstruction_error = float(np.mean(np.abs(features_scaled - reconstruction)))
        
        # Combined Anomaly Score (0-1)
        combined_score = (lstm_anomaly_score + min(reconstruction_error, 1.0)) / 2
        
        # Risk Assessment
        if combined_score > 0.7:
            risk_level = "critical"
            severity = 2
        elif combined_score > 0.4:
            risk_level = "warning"
            severity = 1
        else:
            risk_level = "safe"
            severity = 0
        
        # Identify potential threats
        threats = []
        if data.get("harmonic_distortion", 0) > 5:
            threats.append("High Harmonic Distortion")
        if data.get("frequency", 50) < 49.5 or data.get("frequency", 50) > 50.5:
            threats.append("Frequency Deviation")
        if data.get("voltage", 230) < 200 or data.get("voltage", 230) > 250:
            threats.append("Voltage Out of Range")
        if data.get("power_factor", 1) < 0.85:
            threats.append("Low Power Factor")
        
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "anomaly_detected": combined_score > 0.4,
            "anomaly_score": round(combined_score, 4),
            "lstm_score": round(lstm_anomaly_score, 4),
            "reconstruction_error": round(reconstruction_error, 4),
            "risk_level": risk_level,
            "severity": severity,
            "threats": threats,
            "recommended_action": get_recommendation(risk_level, threats)
        }), 200
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─── Batch Analysis Endpoint ──────────────────────────────────────────────
@app.route("/analyze-batch", methods=["POST"])
def analyze_batch():
    """
    Analyzes multiple data points for trend analysis
    """
    try:
        data = request.json
        
        if not data or "records" not in data:
            return jsonify({"error": "No records provided"}), 400
        
        records = data["records"]
        results = []
        
        for record in records:
            features = np.array([[
                record.get("voltage", 230),
                record.get("current", 45),
                record.get("frequency", 50),
                record.get("power_factor", 0.95),
                record.get("active_power", 10),
                record.get("reactive_power", 2),
                record.get("apparent_power", 10),
                record.get("harmonic_distortion", 3),
                record.get("phase_angle", 10),
                record.get("load_demand", 80),
                record.get("generation_output", 85),
                record.get("net_flow", 5),
                record.get("temperature", 35)
            ]])
            
            features_scaled = scaler.transform(features)
            lstm_pred = lstm_model.predict(features_scaled.reshape(1, 1, -1), verbose=0)
            reconstruction = autoencoder.predict(features_scaled, verbose=0)
            reconstruction_error = float(np.mean(np.abs(features_scaled - reconstruction)))
            
            combined_score = (float(lstm_pred[0][0]) + min(reconstruction_error, 1.0)) / 2
            
            if combined_score > 0.7:
                risk = "critical"
            elif combined_score > 0.4:
                risk = "warning"
            else:
                risk = "safe"
            
            results.append({
                "timestamp": record.get("timestamp", datetime.now().isoformat()),
                "anomaly_score": round(combined_score, 4),
                "risk_level": risk
            })
        
        # Calculate statistics
        scores = [r["anomaly_score"] for r in results]
        
        return jsonify({
            "total_records": len(results),
            "anomalies_detected": sum(1 for r in results if r["anomaly_score"] > 0.4),
            "critical_count": sum(1 for r in results if r["anomaly_score"] > 0.7),
            "average_score": round(np.mean(scores), 4),
            "max_score": round(max(scores), 4),
            "min_score": round(min(scores), 4),
            "trend": "improving" if scores[-1] < np.mean(scores[:-1]) else "degrading",
            "results": results
        }), 200
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ─── Real-time Monitoring Endpoint ──────────────────────────────────────────
@app.route("/status", methods=["GET"])
def get_status():
    """
    Returns current grid status and recommendations
    """
    try:
        return jsonify({
            "grid_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "models": "healthy" if lstm_model is not None else "unhealthy",
                "database": "healthy",
                "api": "responsive"
            },
            "active_alerts": 0,
            "last_check": datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─── Helper Function ──────────────────────────────────────────────────────
def get_recommendation(risk_level, threats):
    """
    Provides actionable recommendations based on risk level
    """
    recommendations = {
        "safe": "Grid operating normally. Continue monitoring.",
        "warning": "Investigate potential issues. Check " + ", ".join(threats) + " immediately.",
        "critical": "URGENT: Activate contingency protocols. Prepare for load shedding if necessary."
    }
    return recommendations.get(risk_level, "Unknown risk level")

# ─── Error Handler ────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
