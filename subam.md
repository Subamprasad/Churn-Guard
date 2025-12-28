# Churn-Guard MLOps Presentation Guide

**Target Audience:** Recruiters, Engineering Managers, and MLOps Professionals.
**Goal:** Demonstrate not just "coding ability" but "Engineering Maturity" and "Business Thinking".

---

## 1. Introduction: The "Why" (The Hook)
**(Video Start - Camera on you)**

"Hi, I'm Subam. In this video, I'm presenting **Churn-Guard**, an end-to-end MLOps system I built to solve customer retention challenges.

Most ML projects fail because they are just 'Jupyter Notebooks thrown over a wall'.
I built Churn-Guard differently. I built it as a **Production System** first.

I solved a specific business problem: **'Predicting Churn to Optimize Net Revenue Retention (NRR)'**."

---

## 2. Problem Framing & Metrics
**(Screen Share: Show `README.md` or Slide)**

"I didn't start with models; I started with the definition.
*   **The Problem:** 'Predict probability of cancellation at the start of the billing cycle.'
*   **The Metric:** I optimized for **Recall**.
    *   *Why?* Because a Missed Churner (False Negative) costs us **$2,000** in Lost Lifetime Value.
    *   A False Alarm (False Positive) only costs a **$10** discount email.
    *   Maximizing Recall creates the best business outcome."

---

## 3. Architecture & Design Patterns (The "Engineering" Part)
**(Screen Share: Show VS Code - `src/` folder)**

"To make this system maintainable and scalable, I utilized software design patterns, replacing simple `if/else` statements with robust architecture.

**A. Strategy Pattern (`src/data_cleaning.py`)**
*(Open file)*
I implemented a `DataStrategy` interface. This allows me to hot-swap pre-processing logic (like `DataPreProcessStrategy` or `DataDivideStrategy`) without breaking the pipeline. It makes experimentation safe and easy.

**B. Factory Pattern (`src/model_dev.py`)**
*(Open file)*
I used a `ModelFactory` to abstract the creation of models. Whether I want `XGBoost`, `LightGBM`, or `RandomForest`, the client code doesn't need to change. I just pass a config name, and the Factory handles the complexity."

---

## 4. The ZenML Pipeline (The "Ops" Part)
**(Screen Share: Show `pipelines/training_pipeline.py` or ZenML Dashboard)**

"For orchestration, I used **ZenML**.
Instead of running a messy script, I defined clear Steps:
*   `ingest_data`
*   `clean_data`
*   `train_model`
*   `evaluate_model`

This pipeline is **Reproducible**. Every run is tracked. I can trace back exactly which data version produced which model version."

---

## 5. Deployment & Serving (The "Product" Part)
**(Screen Share: Show the Web App running at localhost:8000)**

"Finally, a model is useless if it sits on a shelf.
I built a **Flask Inference Engine** (`app.py`) that automatically connects to the ZenML Artifact Store, fetches the latest `best_model`, and serves it via an API.

I also wrapped it in a **Modern, Glassmorphism UI** because user experience matters.

*(Demo: Type in some values - Age: 35, Usage: 100 - and click 'Predict'. Show the result card appearing.)*

This demonstrates the full lifecycle: From **Business Problem** -> **Architecture** -> **Pipeline** -> **Deployment**."

---

## 6. Project Automation
**(Screen Share: Show `run_all.ps1`)**

"To respect developer experience, I completely automated the setup.
I created a `run_all.ps1` script that activates the specific environment (`.venv_fixed` with ZenML 0.93.0), runs the pipeline, and launches the server in a single click."

---

## 7. Conclusion

"Churn-Guard represents my ability to build **Software**, not just scripts. It combines MLOps best practices, strict typing, design patterns, and business-centric problem solving."
