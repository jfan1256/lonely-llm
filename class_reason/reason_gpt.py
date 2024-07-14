import os
import io
import yaml
import base64
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from openai import OpenAI

from utils.system import get_configs

class ReasonGPT:
    # Initialize
    def __init__(self,
                 data_path,
                 output_path
                 ):

        self.data_path = data_path
        self.output_path = output_path

    # Create item list
    def _create_item_list(self):
        # Convert tags to strings
        data = pd.read_csv(self.data_path)
        items = data['body'].values.tolist()
        self.items = items

    @staticmethod
    # Get word count for column
    def _word_count(texts):
        counts = np.zeros(len(texts), dtype=np.int32)
        for i, text in enumerate(texts):
            counts[i] = len(text.split())
        return counts

    # Get gpt topic
    def _gpt_topic(self, item):
        api_key = yaml.safe_load(open(get_configs() / 'api' / 'api.yaml'))['openai']
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Role: {self.ROLE}"
                                    f"Task: Formulate a search query to locate a specific image. Use lowercase and no punctuation."
                                    f"Description: The user might not recall precise details about the image. Craft a query that is sufficiently broad to encompass possible variations, yet detailed enough to facilitate an effective search."
                        },
                    ],
                }
            ],
            temperature=0.75,
            max_tokens=500,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
        )

        summary = response.choices[0].message.content.translate(str.maketrans('', '', '."')).lower()
        return summary

    def _get_gpt_topic(self):
        all_summary = []
        for item in tqdm(self.items, desc='Processing items'):
            summary = self._gpt_topic(item)
            all_summary.append(summary)
        return all_summary

    # Execute parallelized gpt prompt dataframe creation
    def prompt_gpt(self):
        # Create item list
        self._create_item_list()

        # Retrieve reason
        reason = self._get_gpt_topic()

        # Export Dataframe
        data = pd.read_csv(self.data_path)
        data['reason'] = reason

        # Word count
        data['word_count_prompt'] = self._word_count(data['body'].values)
        data['word_count_reason'] = self._word_count(data['reason'].values)

        # Export Good Query
        data.to_csv(f"{self.output_path}.csv", index=False)