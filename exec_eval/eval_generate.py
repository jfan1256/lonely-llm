import warnings
warnings.filterwarnings('ignore')

import yaml
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.print import print_header
from class_models.model_utils import set_seed
from utils.system import get_configs
from class_models.bert_lonely import init_bert_lonely
from class_dataloader.dataloader import Train

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Initialize BertReddit
    print_header("Initialize BertLonely")
    configs = yaml.load(open(get_configs() / 'train' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)
    bert_lonely = init_bert_lonely(pretrained=configs['eval_checkpoint'], configs=configs)
    bert_lonely = bert_lonely.to(device=configs['eval_device'])
    bert_lonely.eval()

    # Initialize dataloader
    print_header("Initialize Dataloader")
    test_data = pd.read_csv(configs['test_path'])
    dataset = Train(data=test_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    # Evaluate
    print_header("Predict")
    label_collect = []
    lonely_loss = []
    sentiment_loss = []
    dice_loss = []
    tversky_loss = []
    constrast_loss = []
    reason_loss = []
    reason_collect = []
    with torch.no_grad():
        for (index, narrative, label, reason) in tqdm(dataloader, desc='Evaluate'):
            label = torch.tensor(label, dtype=torch.float).to(configs['eval_device'])

            # Classify
            labels = bert_lonely.classify(narrative=narrative, num_class=configs['num_class'], device=configs['eval_device'])
            labels = labels.detach().cpu().numpy().tolist()
            label_collect.extend(labels)

            # Generate
            reason = bert_lonely.generate(narrative=narrative, min_length=10, max_length=256, top_p=0.75, temperature=0.75, device=configs['eval_device'])
            reason_collect.extend(reason)

    # Create prediction labels
    if len(reason_collect) < 1:
        eval = pd.DataFrame({'pred_label': label_collect, 'true_label':test_data['label']})
    else:
        eval = pd.DataFrame({'narrative':test_data['narrative'], 'pred_label': label_collect, 'true_label':test_data['label'], 'reason':test_data['reason'], 'pred_reason':reason_collect})
        eval.to_csv(configs['output_dir'] + '/eval.csv', index=False)
    eval['pred_label'] = np.where(eval['pred_label'] > 0.5, 1, 0)

    # Calculate metrics
    print_header("Metrics")
    f1 = f1_score(eval['true_label'], eval['pred_label'], average='binary')
    precision = precision_score(eval['true_label'], eval['pred_label'], average='binary')
    recall = recall_score(eval['true_label'], eval['pred_label'], average='binary')
    print("Loss Lonely:", np.mean(lonely_loss))
    print("Loss Sentiment:", np.mean(sentiment_loss))
    print("Loss Dice:", np.mean(dice_loss))
    print("Loss Tversky:", np.mean(tversky_loss))
    print("Loss Constrastive:", np.mean(constrast_loss))
    print("Loss Reason:", np.mean(reason_loss))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)