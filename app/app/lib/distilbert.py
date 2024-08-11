import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from .classifier import Classifier
from .types import ClassifierConfig


class DistilBERT(Classifier):

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__(config)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "/data/models/distilbert"
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        if not torch.cuda.is_available():
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def run(self, df):

        def infer(title: str):
            encodings = self.tokenizer(
                title, return_tensors="pt", padding=True, truncation=True
            )
            with torch.no_grad():
                output = self.model(**encodings)
            smax = (
                F.softmax(output.logits, dim=1).double().numpy()[0]
            )  # Need [0] because it's a batched output even for single item inference
            return smax

        proba_series = df["title"].apply(lambda title: infer(title))
        return pd.DataFrame(
            proba_series.tolist(),
            columns=["poor", "average", "good"],
        )
