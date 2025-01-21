from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open("heartdiseaseprediction.model", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    msg = None
    if request.method == "POST":
        try:
            # Collect form data
            age = float(request.form["age"])
            chest_pain = float(request.form["r1"])
            bp = float(request.form["BP"])
            cholesterol = float(request.form["CH"])
            max_hr = float(request.form["maxhr"])
            st_depression = float(request.form["STD"])
            vessels_fluro = float(request.form["fluro"])
            thallium = float(request.form["Th"])

            # Prepare data for prediction
            input_features = np.array([[age, chest_pain, bp, cholesterol, max_hr, st_depression, vessels_fluro, thallium]])

            # Make prediction
            prediction = model.predict(input_features)
            msg = "You have heart disease" if prediction[0] == 1 else "No heart disease"
        except Exception as e:
            msg = f"Error: {e}"
    
    return render_template("home.html", msg=msg)

if __name__ == "__main__":
    app.run(debug=True)
