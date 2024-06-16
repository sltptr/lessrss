import json
import os

import numpy as np
from openai import OpenAI

from .classifier import Classifier
from .config import ClassifierConfig


class GPT(Classifier):

    def __init__(self, data: ClassifierConfig) -> None:
        super().__init__(data)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def run(self, df):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """You are an assistant who labels RSS feed items.
                    The user will first give a description of their preferences for RSS items.
                    The user will then give a list of titles for feed items, i.e. ['Post A', 'Post B', 'Post C'].
                    You will then respond with a JSON array of 0s and 1s where 0 means you don't think the user
                    would like the RSS item based on the title, and a 1 means you do think the user would
                    like the item, i.e. [0,1,1] means you think the user wouldn't like Post A but you think
                    that the user would like Post B and Post C. The JSON response should be in the format
                    {labels: []}.""",
                },
                {
                    "role": "user",
                    "content": f"""The following is the description of my preferences for RSS items: {self.prompt}.
                    Here is the list of {df.shape[0]} titles that I need you to process: {df.title.tolist()}.""",
                },
            ],
        )
        content_json = json.loads(completion.choices[0].message.content)
        return np.array(content_json["labels"])
