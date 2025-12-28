Write-Host "========================================"
Write-Host "   Starting Churn-Guard MLOps Flow      "
Write-Host "========================================"

# 1. Activate the correct environment (Python 3.10 / ZenML 0.93.0)
Write-Host "[1/3] Activating Environment (.venv_fixed)..."
& ".\.venv_fixed\Scripts\Activate.ps1"

# 2. Run the ZenML Pipeline
Write-Host "[2/3] Running ZenML Pipeline (Ingest -> Clean -> Train -> Eval)..."
python run.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Pipeline failed! Exiting."
    exit $LASTEXITCODE
}

# 3. Start the Flask Inference App
Write-Host "========================================"
Write-Host "[3/3] Starting Inference App..."
Write-Host "      Go to http://127.0.0.1:8000"
Write-Host "      Press Ctrl+C to stop."
Write-Host "========================================"
python app.py
