from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return "End-to-End Data Science Project API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [[
        data["feature1"],
        data["feature2"],
        data["feature3"],
        data["feature4"]
    ]]

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
