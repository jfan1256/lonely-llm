import warnings
warnings.filterwarnings('ignore')

import os
import yaml
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from utils.print import print_header
from class_models.utils import set_seed
from utils.system import get_configs
from class_models.bert_lonely import init_bert_lonely
from class_dataloader.dataloader_reddit_lonely import RedditLonelyTrain

def read_evaluated_checkpoints(log_path):
    evaluated_checkpoints = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            for line in file:
                if 'checkpoint' in line:
                    parts = line.split()
                    checkpoint_name = parts[0]
                    evaluated_checkpoints.add(checkpoint_name)
    return evaluated_checkpoints

def get_and_remove_best_model(log_path):
    best_model = None
    best_f1 = 0
    lines_to_keep = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if line.startswith('Best model:'):
                parts = line.strip().split()
                best_model = parts[2]
                best_f1 = float(parts[-1])
            else:
                lines_to_keep.append(line)
        with open(log_path, 'w') as file:
            file.writelines(lines_to_keep)
    return best_model, best_f1


if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Get configuration
    configs = yaml.load(open(get_configs() / 'train' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)

    # Initialize Dataloader
    print_header("Initialize Dataloader")
    test_data = pd.read_csv(configs['test_path'])
    dataset = RedditLonelyTrain(data=test_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    # Path to the folder containing checkpoints
    checkpoint_folder = configs['output_dir']

    # Logging setup
    log_file_path = f"{checkpoint_folder}/eval_metric.txt"
    best_model, best_f1 = get_and_remove_best_model(log_file_path)
    log_file = open(log_file_path, "a")
    evaluated_checkpoints = read_evaluated_checkpoints(log_file_path)

    for file in sorted(os.listdir(checkpoint_folder)):
        if file.startswith('checkpoint') and file.endswith('.pth') and file not in evaluated_checkpoints:
            # Load the current checkpoint
            checkpoint_path = os.path.join(checkpoint_folder, file)
            print_header(f"Evaluating {file}")

            # Initialize model with the current checkpoint
            current_configs = configs.copy()
            current_configs['eval_checkpoint'] = checkpoint_path
            bert_lonely = init_bert_lonely(pretrained=current_configs['eval_checkpoint'], configs=current_configs)
            bert_lonely = bert_lonely.to(current_configs['eval_device'])
            bert_lonely.eval()

            # Evaluation loop
            label_collect = []
            lonely_loss = []
            sentiment_loss = []
            dice_loss = []
            tversky_loss = []
            center_loss = []
            angular_loss = []
            constrast_loss = []
            reason_loss = []
            with torch.no_grad():
                for (index, prompt, label, reason) in tqdm(dataloader, desc='Evaluate'):
                    label = torch.tensor(label, dtype=torch.float).to(current_configs['eval_device'])
                    sentiment = bert_lonely.get_sentiment(prompt)
                    sentiment = torch.tensor(sentiment, dtype=torch.float).to(current_configs['eval_device'])

                    loss_binary, loss_sentiment, loss_dice, loss_tversky, loss_center, loss_angular, loss_constrast, loss_reason, labels = bert_lonely(index=index, prompt=prompt, label=label, reason=reason, sentiment=sentiment, device=current_configs['eval_device'])
                    labels = labels.detach().cpu().numpy().tolist()
                    label_collect.extend(labels)
                    lonely_loss.append(loss_binary.item())
                    sentiment_loss.append(loss_sentiment.item())
                    dice_loss.append(loss_dice.item())
                    tversky_loss.append(loss_tversky.item())
                    center_loss.append(loss_center.item())
                    angular_loss.append(loss_angular.item())
                    constrast_loss.append(loss_constrast.item())
                    reason_loss.append(loss_reason.item())

            # Create prediction labels and calculate metrics
            eval = pd.DataFrame({'pred_label': label_collect, 'true_label': test_data['label']})
            eval['pred_label'] = np.where(eval['pred_label'] > 0.5, 1, 0)
            f1 = f1_score(eval['true_label'], eval['pred_label'], average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_model = file

            # Log detailed metrics
            print(f"{file} --> F1 Score: {f1:.4f}  Precision: {precision_score(eval['true_label'], eval['pred_label'], average='binary'):.4f}  Recall: {recall_score(eval['true_label'], eval['pred_label'], average='binary'):.4f}  Loss Lonely: {np.mean(lonely_loss):.4f}  Loss Sentiment: {np.mean(sentiment_loss):.4f}  Loss Dice: {np.mean(dice_loss):.4f}  Loss Tversky: {np.mean(tversky_loss):.4f}  Loss Center: {np.mean(center_loss):.4f}   Loss Angular: {np.mean(angular_loss):.4f}  Loss Constrast: {np.mean(constrast_loss):.4f}  Loss Reason: {np.mean(reason_loss):.4f}", file=log_file, flush=True)

    # Log best model
    print(f"Best model: {best_model} with F1 Score: {best_f1}", file=log_file)
    log_file.close()
