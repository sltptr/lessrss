import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from .classifier import Classifier
from .config import ClassifierConfig

prompt = """You are a classifier for a news website that classifies based on the titles.
The users of this website like content about technology and politics, but they don't like content about sports.
The users also like content about the economy, and they dislike content like listicles, i.e. articles that are just lists of things.
With that in mind, classify the following title as positive or negative: %s
"""


class DistilBERT(Classifier):

    def __init__(self, data: ClassifierConfig) -> None:
        super().__init__(data)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "/models/distilbert"
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def run(self, df):

        def infer(title):
            encodings = self.tokenizer(
                prompt % title,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            encodings = {key: val.to(self.device) for key, val in encodings.items()}
            with torch.no_grad():
                output = self.model(**encodings)
            return torch.argmax(F.softmax(output.logits, dim=1)).item()

        return df["title"].map(infer)
