# 🫀 CVD Risk Detection — AI-Powered Cardiovascular Disease Risk Prediction

A production-grade full-stack machine learning application that predicts cardiovascular disease (CVD) risk using Logistic Regression. Built with FastAPI + React (Vite) + scikit-learn.

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Quick Start (Local)](#quick-start-local)
- [Docker Deployment](#docker-deployment)
- [Backend Deployment](#backend-deployment)
- [Frontend Deployment](#frontend-deployment)
- [API Reference](#api-reference)
- [Example Inputs & Outputs](#example-inputs--outputs)
- [ML Model Details](#ml-model-details)
- [Project Structure](#project-structure)

---

## 🎯 Project Overview

| Feature | Details |
|---------|---------|
| **Model** | Logistic Regression (scikit-learn) |
| **Training Data** | 1,500 clinical records |
| **Accuracy** | 78.7% on held-out test set |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | React 18 + Vite |
| **Containerization** | Docker + Docker Compose |

---

## 🏗 Architecture

```
Browser (React/Vite)
      │
      │  POST /api/predict
      │  GET  /health
      ▼
FastAPI Backend (port 8000)
      │
      │  joblib.load()
      ▼
ML Pipeline
  ├── LabelEncoder   (gender → 0/1)
  ├── StandardScaler (normalize features)
  └── LogisticRegression (predict + predict_proba)
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn

### 1. Clone / Extract the project

```bash
unzip cvd-risk-app.zip
cd cvd-risk-app
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# (Optional) Retrain the model — pre-trained artifacts are already included
python scripts/train_model.py

# Start the API server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API is now running at: http://localhost:8000  
Swagger docs: http://localhost:8000/docs

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env
# Edit .env if your backend runs on a different URL

# Start dev server
npm run dev
```

Frontend is now running at: http://localhost:5173

---

## 🐳 Docker Deployment

### Full Stack with Docker Compose

```bash
# From project root
docker compose up --build

# Backend:  http://localhost:8000
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

Stop containers:
```bash
docker compose down
```

### Build individually

```bash
# Backend only
cd backend
docker build -t cvd-backend .
docker run -p 8000:8000 cvd-backend

# Frontend only
cd frontend
docker build --build-arg VITE_API_URL=http://localhost:8000 -t cvd-frontend .
docker run -p 3000:80 cvd-frontend
```

---

## ☁️ Backend Deployment (e.g. Railway / Render / Fly.io)

1. Set the following environment variables on your platform:
   ```
   FRONTEND_URL=https://your-frontend-domain.com
   PORT=8000
   DEBUG=false
   ```
2. Point the deploy command to: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. Ensure the `app/ml/` directory with `.joblib` files is included in the deployed artifact.

**Render example:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port 10000`

---

## 🌐 Frontend Deployment (e.g. Vercel / Netlify)

1. Set environment variable: `VITE_API_URL=https://your-backend-api.com`
2. Build command: `npm run build`
3. Output directory: `dist`

**Vercel:**
```bash
npm i -g vercel
cd frontend
vercel --prod
```

---

## 🔌 Connecting Frontend with Backend

The frontend reads the API URL from `VITE_API_URL`. Set this at build time or in the `.env` file:

```env
# frontend/.env
VITE_API_URL=http://localhost:8000        # local
VITE_API_URL=https://api.yourdomain.com   # production
```

The backend accepts requests from origins listed in `ALLOWED_ORIGINS`. Update `backend/.env`:

```env
FRONTEND_URL=https://your-frontend.com
```

---

## 📡 API Reference

### `GET /health`

Returns model and service health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "service": "CVD Risk Detection API"
}
```

---

### `POST /api/predict`

**Request Body:**
```json
{
  "age": 55,
  "gender": "Male",
  "cholesterol": 240,
  "blood_pressure": 145,
  "glucose": 110,
  "smoking": 1,
  "alcohol": 0,
  "bmi": 28.5,
  "physical_activity": 0
}
```

**Field Constraints:**

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| age | int | 1–120 | Age in years |
| gender | string | "Male" / "Female" | Biological sex |
| cholesterol | int | 100–400 | Total cholesterol mg/dL |
| blood_pressure | int | 50–250 | Systolic BP mmHg |
| glucose | int | 50–400 | Fasting glucose mg/dL |
| smoking | int | 0 or 1 | 0=No, 1=Yes |
| alcohol | int | 0 or 1 | 0=No, 1=Yes |
| bmi | float | 10.0–60.0 | Body Mass Index |
| physical_activity | int | 0 or 1 | 0=Sedentary, 1=Active |

**Response:**
```json
{
  "prediction": 1,
  "risk_level": "High Risk",
  "risk_probability": 0.7842,
  "risk_percentage": 78.42,
  "confidence": "High",
  "message": "Your cardiovascular risk assessment shows a 78.4% probability...",
  "input_summary": { ... }
}
```

---

## 🧪 Example Inputs & Expected Outputs

### Example 1 — High Risk Patient
```json
Input:
{
  "age": 65, "gender": "Male", "cholesterol": 280,
  "blood_pressure": 160, "glucose": 140,
  "smoking": 1, "alcohol": 1, "bmi": 33.2,
  "physical_activity": 0
}

Expected Output:
{
  "prediction": 1,
  "risk_level": "High Risk",
  "risk_percentage": ~82.5,
  "confidence": "High"
}
```

### Example 2 — Low Risk Patient
```json
Input:
{
  "age": 30, "gender": "Female", "cholesterol": 170,
  "blood_pressure": 115, "glucose": 85,
  "smoking": 0, "alcohol": 0, "bmi": 22.1,
  "physical_activity": 1
}

Expected Output:
{
  "prediction": 0,
  "risk_level": "Low Risk",
  "risk_percentage": ~18.3,
  "confidence": "High"
}
```

---

## 🤖 ML Model Details

### Pipeline
```
Raw Input
  → LabelEncoder (gender: Female=0, Male=1)
  → Feature vector: [age, gender, cholesterol, blood_pressure,
                     glucose, smoking, alcohol, bmi, physical_activity]
  → StandardScaler (zero mean, unit variance)
  → LogisticRegression(class_weight='balanced', max_iter=1000)
  → predict() + predict_proba()
```

### Retrain the Model
```bash
cd backend
python scripts/train_model.py
```
This regenerates all 4 artifacts in `app/ml/`:
- `model.joblib`
- `scaler.joblib`
- `label_encoder.joblib`
- `feature_names.joblib`

---

## 📁 Project Structure

```
cvd-risk-app/
├── backend/
│   ├── app/
│   │   ├── api/routes/
│   │   │   ├── prediction.py     # POST /api/predict
│   │   │   └── health.py         # GET /health
│   │   ├── core/
│   │   │   └── config.py         # Settings + env vars
│   │   ├── ml/
│   │   │   ├── model_loader.py   # Singleton model loader
│   │   │   ├── model.joblib      # Trained model
│   │   │   ├── scaler.joblib     # Feature scaler
│   │   │   ├── label_encoder.joblib
│   │   │   └── feature_names.joblib
│   │   └── schemas/
│   │       └── prediction.py     # Pydantic request/response
│   ├── scripts/
│   │   └── train_model.py        # Retrain script
│   ├── dataset.csv               # Training dataset (1500 records)
│   ├── main.py                   # FastAPI app entry point
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Footer.jsx
│   │   │   └── ResultCard.jsx    # Results display
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Predict.jsx       # Prediction form
│   │   │   ├── About.jsx
│   │   │   └── NotFound.jsx
│   │   ├── hooks/
│   │   │   └── usePrediction.js
│   │   ├── utils/
│   │   │   └── api.js            # Axios API client
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── public/
│   │   └── favicon.svg
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── Dockerfile
│   ├── nginx.conf
│   └── .env.example
│
├── docker-compose.yml
└── README.md
```

---

## ⚕️ Medical Disclaimer

This application is for **educational and demonstration purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for cardiovascular health concerns.

---

*Built with ❤️ using FastAPI · React · scikit-learn · Docker*
