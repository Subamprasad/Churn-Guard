# Churn-Guard AI: MLOps Customer Retention System

**Churn-Guard** is a production-grade MLOps project designed to predict customer churn risk using ZenML pipelines and serve predictions via a modern, glassmorphism-styled Web UI.



## 🚀 Key Features

### 1. **Production-First MLOps Architecture**
Built using **ZenML** to orchestrate reproducible workflows:
- **Ingestion Step**: Handles data loading and synthetic generation.
- **Cleaning Step**: Implements the **Strategy Pattern** for flexible preprocessing.
- **Training Step**: Uses the **Factory Pattern** to switch between XGBoost, LightGBM, and RandomForest.
- **Evaluation Step**: Focuses on **Recall** to minimize missed churners.

### 2. **Advanced Design Patterns**
This project implements robust software engineering patterns to ensure scalability and maintainability:

#### **A. Strategy Pattern**
Used to encapsulate algorithms for specific tasks, allowing them to be interchangeable.
- **Data Cleaning (`src/data_cleaning.py`)**: Defines a `DataStrategy` interface with concrete implementations for `DataPreProcessStrategy` (preprocessing) and `DataDivideStrategy` (splitting data). This allows easily switching between entirely different data handling logic without changing the client code.
- **Model Evaluation (`src/evaluation.py`)**: `Evaluation` abstract class allows different scoring strategies (`MSE`, `Recall`, `F1`).

#### **B. Factory Pattern**
Used to create objects without specifying the exact class of object that will be created.
- **Model Development (`src/model_dev.py`)**: The `ModelFactory` creates instances of model classes (`RandomForestModel`, `XGBoostModel`, `LightGBMModel`) based on a simple string input. This abstracts the complexity of model instantiation from the training step.

### 3. **Premium Web Interface**
A high-aesthetic Flask application (`app.py`) serves the models:
- **Glassmorphism Design**: Modern, translucent UI components.
- **Dynamic Animations**: Smooth transitions and interactive elements.
- **Real-Time Inference**: Fetches the latest trained model artifact directly from the ZenML pipeline.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10
- ZenML

### 1. Clone the Repository
```bash
git clone https://github.com/Subamprasad/Churn-Guard.git
cd Churn-Guard
```

### 2. Run the "One-Click" Setup
We have provided a PowerShell automation script that handles environment activation, training, and serving in one go.

```powershell
.\run_all.ps1
```

**What this script does:**
1.  Activates the dedicated virtual environment (`.venv_fixed`).
2.  Runs the MLOps Training Pipeline (`run.py`).
3.  Starts the Flask Web Server (`app.py`).

### 3. Access the App
Open your browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📂 Project Structure

```text
├── .venv_fixed/        # Python 3.10 Environment (ZenML 0.93.0)
├── pipelines/          # ZenML Pipelines
│   └── training_pipeline.py
├── steps/              # ZenML Pipeline Steps
│   ├── ingest_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   └── evaluate_model.py
├── src/                # Core Logic (Design Patterns)
│   ├── data_cleaning.py  # Strategy Pattern
│   ├── model_dev.py      # Factory Pattern
│   └── evaluation.py     # Strategy Pattern
├── templates/          # HTML Templates
│   └── index.html      # Premium Glassmorphism UI
├── static/             # CSS Assets
│   └── style.css       # Animations & Styling
├── app.py              # Flask Serving Application
├── run.py              # Pipeline Trigger Script
└── run_all.ps1         # Automation Script
```

---

## 🧠 The MLOps Philosophy

This project distinguishes itself by focusing on the **Business Problem** first:
- **Goal**: Optimise Net Revenue Retention (NRR).
- **Metric**: prioritized **Recall** (Cost of False Negative > Cost of False Positive).
- **Guardrails**: Integrated monitoring for feature drift.


