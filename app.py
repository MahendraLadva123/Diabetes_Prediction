from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

with open("diabetes.pkl", "rb") as f:
    pipeline = pickle.load(f)

scaler = pipeline["scaler"]
model = pipeline["model"]

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        user_input = {
            "Pregnancies": int(request.form["Pregnancies"]),
            "Glucose": float(request.form["Glucose"]),
            "BloodPressure": float(request.form["BloodPressure"]),
            "SkinThickness": float(request.form["SkinThickness"]),
            "Insulin": float(request.form["Insulin"]),
            "BMI": float(request.form["BMI"]),
            "DiabetesPedigreeFunction": float(request.form["DiabetesPedigreeFunction"]),
            "Age": int(request.form["Age"])
        }

        new_data = pd.DataFrame([user_input])

        new_data_scaled = scaler.transform(new_data)

        prediction = model.predict(new_data_scaled)[0]

        return render_template("index.html", prediction_text=f"Diabetes Prediction: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
