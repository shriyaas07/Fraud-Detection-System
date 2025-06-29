from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime
import os

# Load model
model = joblib.load('fraud_detection_rf_pipeline.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # --- User Inputs ---
        transaction_amount = float(request.form['transaction_amount'])
        transaction_type = request.form['transaction_type']
        device_type = request.form['device_type']
        is_foreign_transaction = 1 if request.form['is_foreign_transaction'] == 'Yes' else 0
        is_new_location = 1 if request.form['is_new_location'] == 'Yes' else 0

        # Advanced Fields
        failed_logins_24h = int(request.form.get('failed_logins_24h', 0))
        transactions_last_24h = int(request.form.get('transactions_last_24h', 0))
        avg_transaction_amount = float(request.form.get('avg_transaction_amount', 0))
        time_since_last_txn = float(request.form.get('time_since_last_txn', 0))

        # --- Backend-Only Logic ---
        txn_hour = datetime.now().hour
        is_new_device = 1 if device_type.lower() == 'mobile' else 0
        account_age_days = 365  # default, or fetch from user profile
        location_risk = 0.3     # default, or fetch from geo-API

        # --- Construct DataFrame ---
        input_data = pd.DataFrame([{
            'transaction_amount': transaction_amount,
            'transaction_type': transaction_type,
            'device_type': device_type,
            'is_foreign_transaction': is_foreign_transaction,
            'is_new_location': is_new_location,
            'failed_logins_24h': failed_logins_24h,
            'transactions_last_24h': transactions_last_24h,
            'avg_transaction_amount': avg_transaction_amount,
            'time_since_last_txn': time_since_last_txn,
            'txn_hour': txn_hour,
            'is_new_device': is_new_device,
            'account_age_days': account_age_days,
            'location_risk': location_risk
        }])

        # --- Prediction ---
        pred_proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = f"Fraudulent Transaction Detected. Risk Score: {pred_proba:.2f}"
        else:
            result = f"Transaction Looks Safe. Risk Score: {pred_proba:.2f}"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # 10000 is default fallback
    app.run(host="0.0.0.0", port=port, debug=True)

