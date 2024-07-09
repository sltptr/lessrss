import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from .classifier import Classifier
from .config import ClassifierConfig


class DistilBERT(Classifier):

    def __init__(self, data: ClassifierConfig) -> None:
        super().__init__(data)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "/data/models/distilbert"
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def run(self, df):

        def infer(title):
            encodings = self.tokenizer(title, return_tensors="pt")
            output = self.model(**encodings)
            return torch.argmax(F.softmax(output.logits, dim=1)).item()

        return df["title"].map(infer).to_numpy()
