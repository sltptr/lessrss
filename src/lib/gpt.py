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
        titles = {str(i): title for i, title in enumerate(df.title.tolist())}
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an assistant who labels RSS feed items based on user preferences.
                        The user will give you a map of ID's to string titles, an example input might look
                        like {{ '0': 'New science about the Earth', '1': 'Fun ways to spend your summer', '2': 'Top 10 tourist destinations' }}.
                        You must return a JSON map for the ID's to integer labels which are either 0's or 1's.
                        For example, your response should look like {{ '0': 1, '1': 1, '2': 0 }} if you think
                        the user would like the items 0 and 1 but you think the user wouldn't like item 2.
                        For another input, you might respond {{ '0': 0, '1': 1, '2': 0, '3': 1 }} if you think the
                        user would like the items 1 and 3 but wouldn't like the items 0 and 2.
                        To help you label, here is a description of the user's preferences: '{self.prompt}'.
                        """,
                },
                {
                    "role": "user",
                    "content": str(titles),
                },
            ],
        )
        content_json = json.loads(completion.choices[0].message.content)
        print(f"Response from GPT: {content_json}")
        return np.array([content_json[id] for id in sorted(content_json.keys())])
