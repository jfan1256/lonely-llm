import torch

from tqdm import tqdm
from torch.utils.data import Dataset

from class_dataloader.utils import preprocess_text

class RedditLonelyTrain(Dataset):
    def __init__(self,
                 data
                 ):

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        prompt = preprocess_text(self.data.iloc[index]['narrative'])
        label = self.data.iloc[index]['label']
        reason = self.data.iloc[index]['reason']
        # prompt = preprocess_text(self.data.iloc[index]['text'])
        # label = self.data.iloc[index]['overall_label']
        return index, prompt, label, reason

class RedditLonelyMLM(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.examples = []
        self.tokenizer = tokenizer

        for _, row in tqdm(self.data.iterrows(), total=len(self.data), desc='Processing'):
            text = row['body']
            text = preprocess_text(text)
            encoding = self.tokenizer(text, max_length=512, truncation=True, padding="max_length", return_tensors="pt", return_attention_mask=True)
            inputs, labels = self.mask_tokens(encoding.input_ids)
            self.examples.append({
                'input_ids': inputs,
                'labels': labels,
                'attention_mask': encoding['attention_mask']
            })

    def mask_tokens(self, inputs):
        # Create a mask for valid positions which can be masked
        maskable_indices = (inputs != self.tokenizer.cls_token_id) & (inputs != self.tokenizer.sep_token_id) & (inputs != self.tokenizer.pad_token_id)

        # Determine number of tokens to mask: 15% of all maskable tokens
        num_to_mask = int(torch.sum(maskable_indices) * 0.15)
        mask_indices = torch.flatten(maskable_indices.nonzero()).random_(num_to_mask)

        # Expand mask indices to whole words
        all_mask_indices = set()
        for idx in mask_indices:
            whole_word_indices = self.get_whole_word_mask(inputs.squeeze().tolist(), idx.item())
            all_mask_indices.update(whole_word_indices)

        # Create a final mask for the input ids
        final_mask = torch.zeros_like(inputs, dtype=torch.bool)
        for idx in all_mask_indices:
            final_mask[:, idx] = True

        # Apply mask: 80% [MASK], 10% random, 10% unchanged
        true_mask = torch.full_like(inputs, fill_value=-100)  # Default to ignore index
        probability_matrix = torch.rand(inputs.shape)

        # Apply [MASK]
        is_masked = probability_matrix < 0.8
        inputs[is_masked & final_mask] = self.tokenizer.mask_token_id
        true_mask[is_masked & final_mask] = inputs[is_masked & final_mask]

        # Replace with random word
        is_random = (probability_matrix >= 0.8) & (probability_matrix < 0.9)
        random_words = torch.randint(len(self.tokenizer), inputs.shape, dtype=torch.long)
        inputs[is_random & final_mask] = random_words[is_random & final_mask]
        true_mask[is_random & final_mask] = inputs[is_random & final_mask]

        return inputs, true_mask

    def get_whole_word_mask(self, input_ids, index):
        # This function assumes the tokenizer uses BPE or a similar subword tokenization method.
        # It expands the index to include the whole word by checking for subword prefixes.
        token = self.tokenizer.convert_ids_to_tokens(input_ids[index])
        if token.startswith('##'):
            start = index
            while start > 0 and self.tokenizer.convert_ids_to_tokens(input_ids[start - 1]).startswith('##'):
                start -= 1
            end = index
            while end < len(input_ids) - 1 and self.tokenizer.convert_ids_to_tokens(input_ids[end + 1]).startswith('##'):
                end += 1
        else:
            start = index
            end = index
        return range(start, end + 1)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        return {
            'input_ids': item['input_ids'].squeeze(),
            'labels': item['labels'].squeeze(),
            'attention_mask': item['attention_mask'].squeeze()
        }