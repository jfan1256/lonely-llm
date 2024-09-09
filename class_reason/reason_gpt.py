import os
import time
import yaml
import pandas as pd

from tqdm import tqdm
from openai import OpenAI

from utils.system import get_configs
from class_dataloader.utils import preprocess_text

class ReasonGPT:
    def __init__(self, data_path, output_path, prompt):
        self.data_path = data_path
        self.output_path = output_path
        self.prompt = prompt
        self.api_key = yaml.safe_load(open(get_configs() / 'api' / 'api.yaml'))['openai']
        self.client = OpenAI(api_key=self.api_key)

    def _create_item_list(self):
        data = pd.read_csv(self.data_path)
        text = data['narrative'].values.tolist()
        label = data['label'].values.tolist()
        items = list(zip(text, label))
        return items

    def _gpt_topic(self, narrative, label):
        narrative = preprocess_text(narrative)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.prompt}"
                                    f"Narrative: {narrative}\n"
                                    f"Label: {label}"
                        },
                    ],
                }
            ],
            temperature=0.75,
            max_tokens=1000,
            top_p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            seed=1
        )
        summary = response.choices[0].message.content.replace('\n', ' ').lower()
        return summary

    def _process_items(self, items):
        # Check for existing file and read it
        if os.path.exists(self.output_path):
            processed_data = pd.read_csv(self.output_path)
            processed_narratives = set(processed_data['narrative'])
        else:
            processed_data = pd.DataFrame(columns=['narrative', 'label', 'reason'])
            processed_narratives = set()

        # Process new items
        for narrative, label in tqdm(items, desc='Processing items'):
            if narrative not in processed_narratives:
                reason = self._gpt_topic(narrative, label)
                new_data = pd.DataFrame([[narrative, label, reason]], columns=['narrative', 'label', 'reason'])
                processed_data = pd.concat([processed_data, new_data], ignore_index=True)
                processed_data.to_csv(self.output_path, index=False)

    def reason_gpt(self):
        items = self._create_item_list()
        self._process_items(items)