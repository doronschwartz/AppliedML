from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd
import random
app = Flask(__name__)

# Load the preprocessor and XGBoost model
preprocessor = joblib.load('/Users/doronschwartz/YCCS/AppliedML/IEEE-Flask/preprocessor.pkl')
model = joblib.load('/Users/doronschwartz/YCCS/AppliedML/IEEE-Flask/xgboost_model.pkl')

# Define the selected columns and their possible random value ranges
selected_columns = {
    'ProductCD': ['W', 'C', 'R', 'H', 'S'],
    'TransactionAmt': None,  # Random value (not bound)
    'card1': None,  # Random value (not bound)
    'card2': None,  # Random value (not bound)
    'card3': None,  # Random value (not bound)
    'card4': ['visa', 'mastercard', 'discover', 'americanexpress'],
    'card5': None,  # Random value (not bound)
    'card6': ['credit', 'debit', 'chargecard', 'debitofcredit'],
    'P_emaildomain': None,
    'V257': None,  # Random value (not bound)
    'V246': None,  # Random value (not bound)
    'V244': None,  # Random value (not bound)
    'V242': None,  # Random value (not bound)
    'V201': None,  # Random value (not bound)
    'V200': None,  # Random value (not bound)
    'V189': None,  # Random value (not bound)
    'V188': None,  # Random value (not bound)
    'V258': None,  # Random value (not bound)
    'V45': None,  # Random value (not bound)
    'V158': None,  # Random value (not bound)
    'V156': None,  # Random value (not bound)
    'V149': None,  # Random value (not bound)
    'V228': None,  # Random value (not bound)
    'V44': None,  # Random value (not bound)
    'V86': None,  # Random value (not bound)
    'V87': None,  # Random value (not bound)
    'V170': None,  # Random value (not bound)
    'V147': None,  # Random value (not bound)
    'V52': None  # Random value (not bound)
}

# Render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Generate random data for unspecified columns
    data = {
        column: random.choice(values) if values else random.uniform(0, 999999)
        for column, values in selected_columns.items()
    }
    # Add the specified columns from the HTML form
    data.update({
        'card4': request.form['card4'],
        'card6': request.form['card6'],
        'P_emaildomain': request.form['P_emaildomain'],
        'TransactionAmt': float(request.form['TransactionAmt'])
    })

    # Transform and predict
    transformed_data = preprocessor.transform(pd.DataFrame([data]))
    prediction = model.predict_proba(transformed_data)[:, 1]
    return f'Predicted Probability of Fraud: {prediction[0]:.4f}'

if __name__ == '__main__':
    app.run()
