import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd

from tqdm import tqdm
from transformers import BertTokenizer

from class_models.bert import BertConfig, BertModel

if __name__ == '__main__':
    # # LD
    # configs = {'bert_model_checkpoint' : '../paper_experiment/loneliness/corpus_bert_multiloss_0.7tversky_sentiment0.3/checkpoint_02.pth',
    #            'bert_config': '../configs/model/bert_base.json',
    #            'bert_model': 'bert-base-uncased'}
    # Dreaddit
    configs = {'bert_model_checkpoint': '../paper_experiment/dreaddit/corpus_bert_multiloss_0.15tversky_0.9sentiment/checkpoint_04.pth',
                 'bert_config': '../configs/model/bert_base.json',
                 'bert_model': 'bert-base-uncased'}
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]

    # Text Encoder
    bert_config = BertConfig.from_json_file(configs['bert_config'])
    text_encoder = BertModel.from_pretrained(configs['bert_model'], config=bert_config, add_pooling_layer=False, ignore_mismatched_sizes=True)
    checkpoint = torch.load(configs['bert_model_checkpoint'], map_location='cpu')
    model_state_dict = checkpoint['model']
    fixed_state_dict = {key.replace('bert.', ''): value for key, value in model_state_dict.items()}
    text_encoder.load_state_dict(fixed_state_dict, strict=False)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder.eval()

    # Load data
    # # Loneliness
    # file_path = '../paper_experiment/loneliness/best_classifier_decoder_only_all/eval.csv'
    # data = pd.read_csv(file_path)
    # df = pd.DataFrame(data)

    # Dreaddit
    file_path = '../paper_experiment/dreaddit/best_classifier_decoder_only_all/eval.csv'
    data = pd.read_csv(file_path)
    df = pd.DataFrame(data)

    # Set device for computation
    device = "cuda"
    text_encoder = text_encoder.to(device)

    # Define batch size
    batch_size = 32

    # Process in batches
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i + batch_size]
        texts1 = tokenizer(batch['reason'].tolist(), return_tensors="pt", padding='max_length', truncation=True).to(device)
        texts2 = tokenizer(batch['pred_reason'].tolist(), return_tensors="pt", padding='max_length', truncation=True).to(device)

        with torch.no_grad():
            outputs1 = text_encoder(texts1.input_ids, attention_mask=texts1.attention_mask, return_dict=True, mode='text')
            outputs2 = text_encoder(texts2.input_ids, attention_mask=texts2.attention_mask, return_dict=True, mode='text')

        embed1 = outputs1.last_hidden_state[:, 0]
        embed2 = outputs2.last_hidden_state[:, 0]

        cos_sim = torch.cosine_similarity(embed1, embed2).cpu().numpy()
        results.extend(cos_sim)

    # Add results to DataFrame
    df['cos_sim'] = results
    print('mean cos sim score: {:2f}'.format(df['cos_sim'].mean()))
    print('std cos sim score: {:2f}'.format(df['cos_sim'].std()))
    print('max cos sim score: {:2f}'.format(df['cos_sim'].max()))
    print('min cos sim score: {:2f}'.format(df['cos_sim'].min()))