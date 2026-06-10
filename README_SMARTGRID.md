# SmartGrid Security — Deep Learning Anomaly Detection

A comprehensive full-stack application for detecting anomalies in power grid operations using deep learning techniques (LSTM + Autoencoder ensemble).

## 🎯 Project Overview

**SmartGrid Security** enhances power grid security by using deep learning models to detect anomalies and potential threats in real-time. The system monitors critical grid parameters and identifies deviations that could indicate:

- Equipment failures
- Network attacks
- Load imbalances
- Frequency deviations
- Voltage anomalies
- Harmonic distortions

## 🏗️ Architecture

### Backend (Flask + TensorFlow)
- **LSTM Model**: Temporal anomaly detection using Long Short-Term Memory networks
- **Autoencoder**: Reconstruction-based anomaly scoring
- **Ensemble Approach**: Combines both models for robust detection
- **REST API**: Exposes `/detect-anomaly` and `/analyze-batch` endpoints
- **Real-time Processing**: Handles continuous grid monitoring

### Frontend (HTML5 + CSS3 + JavaScript)
- **Modern Dashboard**: Real-time metrics and risk assessment
- **13 Grid Parameters**: Voltage, Current, Frequency, Power Factor, etc.
- **Risk Level Badges**: Safe, Warning, Critical classifications
- **Threat Detection**: Lists identified grid anomalies
- **Responsive Design**: Works on desktop and mobile devices

## 📊 Monitored Grid Features

1. **Voltage (V)**: Main supply voltage level
2. **Current (A)**: Current flow in amperes
3. **Frequency (Hz)**: Power grid frequency (nominal: 50Hz)
4. **Power Factor**: Efficiency of power usage
5. **Active Power (kW)**: Real power consumption
6. **Reactive Power (kVAR)**: Reactive power component
7. **Apparent Power (kVA)**: Total power consumption
8. **Harmonic Distortion (%)**: Higher-frequency signal contamination
9. **Phase Angle (°)**: Phase relationship between voltage and current
10. **Load Demand (%)**: Percentage of grid capacity in use
11. **Generation Output (%)**: Current generation capacity
12. **Net Flow (MW)**: Power flowing between nodes
13. **Temperature (°C)**: Ambient temperature affecting equipment

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.13+
- Flask 2.3+
- Modern web browser

### 1. Environment Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r ../requirements_smartgrid.txt
```

### 2. Train Deep Learning Models

```bash
# From project root
python model_training_smartgrid.py
```

This will:
- Generate 5000 synthetic SmartGrid data samples (80% normal, 20% anomalies)
- Train LSTM model for temporal anomaly detection
- Train Autoencoder for reconstruction-based detection
- Save models to `backend/models/`

### 3. Start Backend Server

```bash
# From backend directory (with venv activated)
python app_smartgrid.py
```

Server starts at `http://localhost:5000`

### 4. Open Frontend

```bash
# Open in your browser
smartgrid.html
```

Or directly: `file:///path/to/smartgrid.html`

## 📡 API Endpoints

### Health Check
```
GET /
```

Response:
```json
{
  "service": "SmartGrid Security System",
  "status": "operational",
  "models_loaded": true
}
```

### Detect Single Anomaly
```
POST /detect-anomaly
```

Request:
```json
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
```

Response:
```json
{
  "timestamp": "2024-06-10T14:30:45.123456",
  "anomaly_detected": false,
  "anomaly_score": 0.2345,
  "lstm_score": 0.1234,
  "reconstruction_error": 0.3456,
  "risk_level": "safe",
  "severity": 0,
  "threats": [],
  "recommended_action": "Grid operating normally. Continue monitoring."
}
```

### Batch Analysis
```
POST /analyze-batch
```

Request:
```json
{
  "records": [
    {"voltage": 230, "current": 45, ...},
    {"voltage": 225, "current": 48, ...}
  ]
}
```

Response includes statistics, trends, and per-record anomaly scores.

## 🎨 Dashboard Features

### Real-time Metrics
- **Anomaly Score**: Combined model prediction (0-1 scale)
- **Risk Level**: Safe, Warning, or Critical
- **LSTM Score**: Individual LSTM prediction
- **Reconstruction Error**: Autoencoder MSE value

### Interactive Analysis
- Enter grid parameters in real-time
- Instant anomaly detection
- Visual threat indicators
- Actionable recommendations

### Threat Detection
Automatically identifies:
- High harmonic distortion
- Frequency deviations
- Voltage out of range
- Low power factor
- Load/generation imbalances

## 🧠 Deep Learning Models

### LSTM Model
- **Architecture**: 2-layer LSTM with dropout
- **Input**: 10-step sequences of 13 features each
- **Output**: Anomaly probability (0-1)
- **Purpose**: Captures temporal patterns in grid data

### Autoencoder Model
- **Architecture**: 3-layer encoder-decoder
- **Bottleneck**: 5-dimensional latent representation
- **Loss Function**: Mean Squared Error (Reconstruction Error)
- **Purpose**: Detects deviations from normal operating patterns

### Ensemble Scoring
```
Combined Score = (LSTM Score + min(Reconstruction Error, 1.0)) / 2
```

**Risk Classification**:
- **Safe**: Score < 0.4
- **Warning**: 0.4 ≤ Score ≤ 0.7
- **Critical**: Score > 0.7

## 📈 Model Training Results

Expected performance on test set:
- LSTM Accuracy: ~85-90%
- Autoencoder MSE: ~0.05-0.08
- Combined Detection Rate: ~88%

## 🔧 Configuration

### Backend Config (app_smartgrid.py)
- API runs on `0.0.0.0:5000`
- CORS enabled for frontend requests
- Models loaded on startup

### Model Training Config (model_training_smartgrid.py)
- Sequence length: 10 timesteps
- Feature dimension: 13
- Epochs: 50
- Batch size: 32
- Validation split: 20%

## 📦 Project Structure

```
project2/
├── backend/
│   ├── app_smartgrid.py           # Flask backend
│   ├── models/
│   │   ├── lstm_anomaly_detector.h5
│   │   ├── autoencoder.h5
│   │   └── scaler.pkl
│   └── requirements_smartgrid.txt
├── frontend/
│   └── smartgrid.html              # Web dashboard
├── model_training_smartgrid.py      # Model training script
└── README_SMARTGRID.md              # This file
```

## 🛡️ Security Features

- CORS protection for API
- Data normalization prevents injection
- Ensemble detection reduces false positives
- Real-time threat identification
- Audit-ready logging

## 🚨 Example Anomalies

### Scenario 1: Voltage Spike
```json
{"voltage": 280, "current": 45, ...}
→ Risk Level: Critical
→ Threat: Voltage Out of Range
```

### Scenario 2: High Harmonic Distortion
```json
{"harmonic_distortion": 8.5, ...}
→ Risk Level: Warning
→ Threat: High Harmonic Distortion
```

### Scenario 3: Frequency Deviation
```json
{"frequency": 48.9, ...}
→ Risk Level: Warning
→ Threat: Frequency Deviation
```

## 🔄 Deployment

### Local Development
```bash
python app_smartgrid.py
```

### Production (Gunicorn)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app_smartgrid:app
```

### Docker (Optional)
Create `Dockerfile` for containerization:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements_smartgrid.txt .
RUN pip install -r requirements_smartgrid.txt
COPY backend/ .
CMD ["python", "app_smartgrid.py"]
```

## 📚 Technologies Used

- **Backend**: Flask, TensorFlow, Keras, Scikit-learn, Pandas, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **ML Models**: LSTM, Autoencoder
- **Deployment**: Gunicorn, Docker (optional)

## 🎓 Learning Resources

- [TensorFlow LSTM Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Autoencoder Anomaly Detection](https://www.tensorflow.org/tutorials/generative/autoencoder)
- [Time Series Anomaly Detection](https://towardsdatascience.com/time-series-anomaly-detection-with-lstm-autoencoders-e4f47f6f50ff)

## 🤝 Contributing

Feel free to enhance the system with:
- Additional ML models (Isolation Forest, One-Class SVM)
- Real grid data integration
- Advanced visualization (Plotly, Chart.js)
- Database integration (PostgreSQL)
- Mobile app (React Native)

## 📄 License

This project is for educational and research purposes.

## 👨‍💻 Author

SmartGrid Security System — Enhancing Power Grid Resilience with Deep Learning

---

**Last Updated**: June 2024
**Version**: 1.0.0
