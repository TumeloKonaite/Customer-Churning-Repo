from flask import Flask, request, render_template, jsonify
from datetime import datetime
import json
import os

from src.decisioning import (
    ACTION_COSTS,
    estimate_clv,
    expected_net_gain,
    recommended_action,
)
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Load model metadata
def load_metadata():
    try:
        with open('artifacts/metadata.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"training_date": "unknown", "model_name": "churn_predictor"}

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': True,
        'metadata': load_metadata()
    })

# API endpoint for predictions
@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        
        # Create CustomData instance
        data_instance = CustomData(
            credit_score=float(data.get('CreditScore')),
            geography=data.get('Geography'),
            gender=data.get('Gender'),
            age=float(data.get('Age')),
            tenure=float(data.get('Tenure')),
            balance=float(data.get('Balance')),
            num_of_products=float(data.get('NumOfProducts')),
            has_cr_card=float(data.get('HasCrCard')),
            is_active_member=float(data.get('IsActiveMember')),
            estimated_salary=float(data.get('EstimatedSalary'))
        )

        pred_df = data_instance.get_data_as_data_frame()
        pipeline = PredictPipeline()
        result, proba = pipeline.predict(pred_df)

        # Calculate business metrics
        churn_probability = float(proba[0]) if proba is not None else None
        clv = estimate_clv({
            "Balance": data.get('Balance'),
            "Tenure": data.get('Tenure'),
            "EstimatedSalary": data.get('EstimatedSalary')
        })
        action = recommended_action(churn_probability)
        action_cost = ACTION_COSTS.get(action, 0.0)
        net_gain = expected_net_gain(churn_probability, clv, action_cost)

        metadata = load_metadata()
        
        return jsonify({
            'status': 'success',
            'p_churn': churn_probability,
            'predicted_label': int(result[0]),
            'clv': clv,
            'recommended_action': action,
            'net_gain': net_gain,
            'model_name': metadata.get('model_name'),
            'model_version': metadata.get('version', '1.0.0'),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions via web form
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            credit_score = float(request.form.get('CreditScore'))
            geography = request.form.get('Geography')
            gender = request.form.get('Gender')
            age = float(request.form.get('Age'))
            tenure = float(request.form.get('Tenure'))
            balance = float(request.form.get('Balance'))
            num_of_products = float(request.form.get('NumOfProducts'))
            has_cr_card = float(request.form.get('HasCrCard'))
            is_active_member = float(request.form.get('IsActiveMember'))
            estimated_salary = float(request.form.get('EstimatedSalary'))

            data = CustomData(
                credit_score=credit_score,
                geography=geography,
                gender=gender,
                age=age,
                tenure=tenure,
                balance=balance,
                num_of_products=num_of_products,
                has_cr_card=has_cr_card,
                is_active_member=is_active_member,
                estimated_salary=estimated_salary,
            )

            pred_df = data.get_data_as_data_frame()
            pipeline = PredictPipeline()
            result, proba = pipeline.predict(pred_df)

            if result[0] == 1:
                pred_text = "This customer is at high risk of leaving. Immediate retention actions are recommended."
            else:
                pred_text = "This customer is likely to stay. No urgent retention action needed."

            churn_probability = None
            clv = None
            action = None
            net_gain = None

            if proba is not None:
                churn_probability = float(proba[0])
                clv = estimate_clv(
                    {
                        "Balance": balance,
                        "Tenure": tenure,
                        "EstimatedSalary": estimated_salary,
                    }
                )
                action = recommended_action(churn_probability)
                action_cost = ACTION_COSTS.get(action, 0.0)
                net_gain = expected_net_gain(churn_probability, clv, action_cost)
                pred_text = f"{pred_text} (Churn probability: {churn_probability:.2%})"

            return render_template(
                'home.html',
                results=pred_text,
                churn_probability=churn_probability,
                clv=clv,
                action=action,
                net_gain=net_gain,
            )

        except Exception as e:
            return render_template(
                'home.html',
                error=f"Error processing request: {str(e)}",
                results=None,
                churn_probability=None,
                clv=None,
                action=None,
                net_gain=None,
            )

    return render_template(
        'home.html',
        results=None,
        churn_probability=None,
        clv=None,
        action=None,
        net_gain=None,
    )

if __name__ == "__main__":
    # Change port to 5000 to match Docker configuration
    app.run(host="0.0.0.0", port=5000, debug=True)
