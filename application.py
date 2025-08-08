from flask import Flask, request, jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


application = Flask(__name__)
app=application




## import random regressor and standard scaler pickle
Random_model = pickle.load(open('models/RandomClassifier.pkl', 'rb'))
Standard_Scaler = pickle.load(open('models/sc.pkl', 'rb'))


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')


## Route for home page
@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':

        CreditScore = float(request.form.get('CreditScore'))
        Geography = request.form.get('Geography')
        Gender = request.form.get('Gender')
        Age = float(request.form.get('Age'))
        Tenure = float(request.form.get('Tenure'))
        Balance = float(request.form.get('Balance'))
        NumOfProducts = float(request.form.get('NumOfProducts'))
        HasCrCard = float(request.form.get('HasCrCard'))
        IsActiveMember = float(request.form.get('IsActiveMember'))
        EstimatedSalary = float(request.form.get('EstimatedSalary'))

        # Encode categorical variables manually (like-for-like, not enhanced)
        Gender = 1 if Gender == 'Male' else 0
        geo_France = 1 if Geography == 'France' else 0
        geo_Germany = 1 if Geography == 'Germany' else 0
        geo_Spain = 1 if Geography == 'Spain' else 0

        # Form the input vector
        new_data_scaled = Standard_Scaler.transform([[geo_France, geo_Germany, geo_Spain, CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]])

        result = Random_model.predict(new_data_scaled)

        if result[0] == 1:
            pred_text = "This customer is at high risk of leaving. Immediate retention actions are recommended."
        else:
            pred_text = "This customer is likely to stay. No urgent retention action needed."

        return render_template('home.html', results=pred_text)


    else:
        return render_template('home.html')
























if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

