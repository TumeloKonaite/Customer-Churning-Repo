from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app=application


## Route for home page
@app.route('/')
def index():
    return render_template('index.html')


## Route for home page
@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':

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

        if proba is not None:
            pred_text = f"{pred_text} (Churn probability: {proba[0]:.2%})"

        return render_template('home.html', results=pred_text)


    else:
        return render_template('home.html')
























if __name__=="__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
