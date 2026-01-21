from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load Model and Encoders
MODEL_PATH = os.path.join("model", "titanic_survival_model.pkl")
LE_SEX_PATH = os.path.join("model", "le_sex.pkl")

model = joblib.load(MODEL_PATH)
le_sex = joblib.load(LE_SEX_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get input values
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            fare = float(request.form['fare'])

            # Create DataFrame
            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'Fare': [fare],
                'SibSp': [sibsp]
            })
            
            # Use the 5 selected features
            features_for_model = input_data[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp']].copy()

            # Preprocess
            features_for_model['Sex'] = le_sex.transform(features_for_model['Sex'])

            # Predict
            prediction = model.predict(features_for_model)
            result = "Survived" if prediction[0] == 1 else "Did Not Survive"

            return render_template('index.html', result=result, 
                                   pclass=pclass, sex=sex, age=age, fare=fare, sibsp=sibsp)
        except Exception as e:
            return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
