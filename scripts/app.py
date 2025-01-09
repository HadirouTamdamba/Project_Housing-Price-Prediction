from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger le modèle
model = joblib.load("models/best_housing_price_model.pkl") 

@app.route("/", methods=["POST"])
def home():
    return "Welcome to the Housing Price Prediction API! Use /predict to get predictions."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400  

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0") 
