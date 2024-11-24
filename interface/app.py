import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    if prediction[0] > 6:
        prediction_text = "The quality of wine is good."
    else:
        prediction_text = "The quality of wine is bad."
            
    return render_template("result.html", prediction_text=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)