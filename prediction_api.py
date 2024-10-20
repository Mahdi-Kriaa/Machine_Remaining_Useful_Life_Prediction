from flask import Flask, jsonify, request, render_template
from src.pipelines.prediction_pipeline import prediction_pipeline
import pandas as pd
from src.exception import CustomException
import sys

app = Flask(__name__)

@app.route("/prediction_api", methods=["POST"])
def prediction_api():
    try:
        data = request.get_json(force=True)
        data = pd.DataFrame(data)
        predict_pipeline = prediction_pipeline()
        result = predict_pipeline.predict(data)
        pred = dict(prediction=float(result[0][0]))
        return jsonify(pred)
    except Exception as e:
        raise CustomException(e, sys)

if __name__=="__main__":
    app.run(host="0.0.0.0", port=4000, debug=True)

