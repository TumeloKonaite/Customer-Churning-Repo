# Customer Churn Prediction Web App

## 📌 Overview
This is a **Flask-based web application** for predicting customer churn using a pre-trained machine learning model.  
Users can input customer details, and the app will predict whether the customer is **at high risk of leaving** or **likely to stay**.  
The model is powered by **scikit-learn** and uses a **RandomForest Classifier** (saved as `RandomClassifier.pkl`) along with a `StandardScaler` for data preprocessing.

---

## 🛠 Features
- **User-friendly web interface** for entering customer data.
- **Real-time churn prediction** based on multiple customer features.
- **Board-level insights**:
  - High risk → recommends retention actions.
  - Low risk → indicates no urgent intervention required.
- **Custom encoding & scaling** to match the model’s training process.

---

## 📂 Project Structure
