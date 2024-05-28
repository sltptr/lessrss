import sys

import joblib
import yaml
from flask import Flask, jsonify, request

sys.path.append("/app/data/")

from custom_transformer import DenseTransformer

app = Flask(__name__)


def load_config():
    with open("/app/data/config.yml", "r") as file:
        config = yaml.safe_load(file)
    return config


def load_tf_idf_logistic():
    return joblib.load("/app/data/tf-idf-logreg.joblib")


def load_tf_idf_naive_bayes():
    return joblib.load("/app/data/tf-idf-nb.joblib")


def load_fasttext():
    return


def load_distilbert():
    return


model_fn_map = {
    "tf-idf-logreg": load_tf_idf_logistic,
    "tf-idf-nb": load_tf_idf_naive_bayes,
}


def load_models_with_scalars(config):
    scalars = config.get("scalars", {})
    models = [
        {"model": model_fn_map[key](), "scalar": value}
        for key, value in scalars.items()
    ]
    return models


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "...")
    predictions = []
    for model_dict in load_models_with_scalars(load_config()):
        model, scalar = model_dict["model"], model_dict["scalar"]
        predictions.append(model.predict([text])[0] * scalar)
    return jsonify({"predictions": predictions})


# Get RSS feed
@app.route("/rss", methods=["GET"])
def rss():
    return "todo"


# Update RSS feed
@app.route("/update", methods=["GET"])
def update():
    return "todo"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
