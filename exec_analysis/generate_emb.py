import warnings
warnings.filterwarnings('ignore')

import torch
import brotli
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer

from utils.system import get_data
from class_models.bert import BertConfig, BertModel


# Save using Brotli compression
def save_brotli(data, filename):
    with open(filename, 'wb') as f:
        compressed_data = brotli.compress(pickle.dumps(data))
        f.write(compressed_data)

# Extract [CLS] Token Embedding
def extract_cls_token_emb(df, model, tokenizer, text_column, batch_size=100, max_length=512):
    # Set device for computation
    features = []
    model.eval()
    device = "cuda"
    model = model.to(device)

    # Iterate through text
    with (torch.no_grad()):
        for i in tqdm(range(0, len(df), batch_size)):
            # Create batch
            batch = df.iloc[i:i + batch_size]
            texts = batch[text_column].tolist()
            tokenized_input = tokenizer(texts, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt').to(device)
            tokenized_output = model(tokenized_input.input_ids, attention_mask=tokenized_input.attention_mask, return_dict=True, mode='text')

            # Convert to Numpy
            batch_features = tokenized_output.last_hidden_state[:, 0].cpu().numpy()
            features.append(batch_features)

    # Concatenate all batches
    return np.concatenate(features, axis=0)

if __name__ == '__main__':
    # Load the CSV file into a DataFrame
    train_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-train-1.csv"
    test_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-test-1.csv"
    val_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-val-1.csv"
    # Load the data
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    eval_df = pd.read_csv(val_file_path)

    # Configs
    configs = {'bert_model_checkpoint': 'checkpoint_used_for_analysis/bert_corpus_checkpoint_30.pth',
                'bert_config': '../configs/model/bert_base.json',
                'bert_model': 'bert-base-uncased'}
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

    # Model
    bert_config = BertConfig.from_json_file(configs['bert_config'])
    model = BertModel.from_pretrained(configs['bert_model'], config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
    checkpoint = torch.load(configs['bert_model_checkpoint'], map_location='cpu')
    model_state_dict = checkpoint['model']
    fixed_state_dict = {key.replace('bert.', ''): value for key, value in model_state_dict.items()}
    model.load_state_dict(fixed_state_dict, strict=False)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Generate Features
    print("Generate Features for Training Dataset...")
    train_features = extract_cls_token_emb(train_df, model, tokenizer, text_column='narrative')
    save_brotli(train_features, 'features/train_features.brotli')

    print("Generate Features for Test Dataset...")
    test_features = extract_cls_token_emb(test_df, model, tokenizer, text_column='narrative')
    save_brotli(test_features, 'features/test_features.brotli')

    print("Generate Features for Eval Dataset...")
    eval_features = extract_cls_token_emb(eval_df, model, tokenizer, text_column='narrative')
    save_brotli(eval_features, 'features/eval_features.brotli')
    print("Features Generation done...")