from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from zenml.client import Client

app = Flask(__name__)

def get_latest_model():
    """Fetch the latest model artifact from ZenML."""
    client = Client()
    # Fetch the latest run of the training pipeline
    try:
        pipeline = client.get_pipeline("train_pipeline")
        last_run = pipeline.last_run
        
        # The step 'train_model' outputs the model. 
        # Check if the step status is completed
        if last_run.status == "completed":
            # The output name is usually just the return type name or based on signature
            # We can iterate outputs or get by name if we know it.
            # In our step definition, we didn't name outputs explicitly, so it's likely 'output' or index 0.
            train_step = last_run.steps["train_model"]
            # ZenML v0.93+: steps have 'outputs' dictionary
            # Let's try to get the first output
            for output_name, artifact_version in train_step.outputs.items():
                # Sometimes it returns a list of artifacts
                if isinstance(artifact_version, list):
                    return artifact_version[0].load()
                return artifact_version.load()
            
        print("Pipeline run not completed or found.")
        return None
    except Exception as e:
        print(f"Error loading model from pipeline: {e}")
        return None

# Load model at startup
model = get_latest_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded properly. Check ZenML connection."}), 500

    try:
        # Get data from form
        data = request.form
        
        # Create DataFrame (matches training schema)
        input_data = pd.DataFrame([{
            'age': float(data.get('age')),
            'usage_minutes': float(data.get('usage_minutes')),
            'contract_length': float(data.get('contract_length')),
            'support_calls': float(data.get('support_calls')),
            'payment_delay': float(data.get('payment_delay'))
        }])
        
        # Simple Preprocessing (matching logic in clean_data.py but simplified for demo)
        # Note: In production, you'd pickle the fitted 'DataStrategy' or pipeline.
        # Here we just ensure types.
        
        # Predict
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        result = "Churn Risk: HIGH" if prediction == 1 else "Churn Risk: LOW"
        
        return jsonify({
            "prediction": result,
            "probability": f"{probability:.2f}",
            "raw_pred": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=8000)
