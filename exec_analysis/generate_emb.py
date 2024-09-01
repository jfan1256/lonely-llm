import warnings
warnings.filterwarnings('ignore')
import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import pandas as pd
from utils.system import get_data
import brotli
import pickle
from class_models.bert import BertConfig, BertModel
from tqdm import tqdm

###Load data file
# Load the CSV file into a DataFrame
train_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-train-1.csv"
test_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-test-1.csv"
val_file_path = get_data() / "loneliness" / "OurLabeledData" / "SamplingData-1" / "loneliness-val-1.csv"
# Load the data
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)
eval_df = pd.read_csv(val_file_path)


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

###Use the Model for Feature Extraction
# Function to tokenize and extract features
def extract_features_from_text(df, model, tokenizer, text_column, batch_size=100, max_length=512):
    # Set device for computation
    features = []
    model.eval()  # Ensure the model is in evaluation mode
    device = "cuda"
    model = model.to(device)

    with (torch.no_grad()):
        for i in tqdm(range(0, len(df), batch_size)):
            batch = df.iloc[i:i + batch_size]
            texts = batch[text_column].tolist()
            tokenized_input = tokenizer(
                texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            ).to(device)
            # Move inputs to the same device as the model
            #tokenized_input = {key: val.to(model.device) for key, val in tokenized_input.items()}
            tokenized_output = model(tokenized_input.input_ids, attention_mask=tokenized_input.attention_mask, return_dict=True, mode='text')

            # Move the output to the CPU before converting to a NumPy array
            batch_features = tokenized_output.last_hidden_state[:, 0].cpu().numpy()
            features.append(batch_features)
    # Concatenate all batches
    return np.concatenate(features, axis=0)

# Save function using Brotli compression
def save_brotli(data, filename):
    with open(filename, 'wb') as f:
        compressed_data = brotli.compress(pickle.dumps(data))
        f.write(compressed_data)

###Generate Features
print("Generate Features for Training Dataset...")
train_features = extract_features_from_text(train_df, model, tokenizer, text_column='narrative')
save_brotli(train_features, 'features/train_features.brotli')

print("Generate Features for Test Dataset...")
test_features = extract_features_from_text(test_df, model, tokenizer, text_column='narrative')
save_brotli(test_features, 'features/test_features.brotli')

print("Generate Features for Eval Dataset...")
eval_features = extract_features_from_text(eval_df, model, tokenizer, text_column='narrative')
save_brotli(eval_features, 'features/eval_features.brotli')

print("Features Generation done...")