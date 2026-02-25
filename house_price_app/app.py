from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load Model

model = joblib.load("model/pipeline.pkl")


# Home Route

@app.route("/", methods=["GET", "POST"])
def home():
  prediction = None
  error = None

  if request.method == "POST":
    try:
      area = float(request.form["area"])
      bedrooms = int(request.form["bedrooms"])
      age = float(request.form["age"])

      # Basic Validation
      if area <= 0 or bedrooms <= 0 or age < 0:
        error = "Please enter valid positive values."
      else:
        features = np.array([[area, bedrooms, age]])
        result = model.predict(features)
        prediction = round(result[0], 2)

    except ValueError:
      error = "Invalid input. Please enter numeric values."

  return render_template("index.html", prediction=prediction, error=error)


# Run Application

if __name__ == "__main__":
  app.run(debug=True)