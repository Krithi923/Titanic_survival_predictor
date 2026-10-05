"""
Flask API for the Titanic Survival Predictor.
Loads model.pkl and exposes a /predict endpoint.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)   # allow frontend to call this API from another origin

# ---- Load trained model ----
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Titanic Survival Predictor API is running",
        "endpoints": {
            "POST /predict": "Predict survival probability"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract & validate inputs
        pclass = int(data["pclass"])
        sex    = 1 if data["sex"].lower() == "male" else 0
        age    = float(data["age"])
        sibsp  = int(data["sibsp"])
        parch  = int(data["parch"])

        # Build feature array in the same order used in training:
        # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
        features = np.array([[pclass, sex, age, sibsp, parch]])

        # Predict
        prob = model.predict_proba(features)[0][1]   # probability of survival
        pred = int(model.predict(features)[0])

        return jsonify({
            "success": True,
            "survived": bool(pred),
            "probability": round(float(prob) * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)