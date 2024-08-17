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
from utils.system import get_configs
from class_models.model_utils import set_seed
from class_dataloader.dataloader import Train
from class_models.bert_lonely import init_bert_lonely

# Read evaluated checkpoints
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

# Get and remove best_model from eval_metric.txt
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

    # Get configs
    configs = yaml.load(open(get_configs() / 'train' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)

    # Load in test data
    print_header("Initialize Dataloader")
    test_data = pd.read_csv(configs['test_path'])
    dataset = Train(data=test_data)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)

    # Set checkpoints
    checkpoint_folder = configs['output_dir']
    log_file_path = f"{checkpoint_folder}/eval_metric.txt"
    best_model, best_f1 = get_and_remove_best_model(log_file_path)
    log_file = open(log_file_path, "a")
    evaluated_checkpoints = read_evaluated_checkpoints(log_file_path)

    # Evaluate checkpoints
    for file in sorted(os.listdir(checkpoint_folder)):
        if file.startswith('checkpoint') and file.endswith('.pth') and file not in evaluated_checkpoints:
            # Read in checkpoint
            checkpoint_path = os.path.join(checkpoint_folder, file)
            print_header(f"Evaluating {file}")
            current_configs = configs.copy()
            current_configs['eval_checkpoint'] = checkpoint_path
            model = init_bert_lonely(pretrained=current_configs['eval_checkpoint'], configs=current_configs)
            model = model.to(current_configs['eval_device'])
            model.eval()

            # Evaluate model
            loss_collect = {}
            label_collect = []
            with torch.no_grad():
                for (index, narrative, label, reason) in tqdm(dataloader, desc='Evaluate'):
                    # Get label and sentiment
                    label = torch.tensor(label, dtype=torch.float).to(current_configs['eval_device'])
                    sentiment = model.get_sentiment(narrative)
                    sentiment = torch.tensor(sentiment, dtype=torch.float).to(current_configs['eval_device'])

                    # Get losses and probabilities
                    results = model(index=index, narrative=narrative, label=label, reason=reason, sentiment=sentiment, device=current_configs['eval_device'])
                    probs = model.classify(narrative=narrative, num_class=current_configs['num_class'], device=current_configs['eval_device']).detach().cpu().numpy().tolist()
                    if current_configs['num_class'] == 2:
                        probs = np.array(probs)
                        preds = (probs > 0.5).astype(int).tolist()
                    else:
                        preds = np.argmax(probs, axis=1).tolist()
                    label_collect.extend(preds)
                    for key, value in results.items():
                        if key not in loss_collect:
                            loss_collect[key] = []
                        loss_collect[key].append(value.item())

            # Calculate F1
            if current_configs['num_class'] == 2:
                average_method = 'binary'
            else:
                average_method = 'macro'
            eval = pd.DataFrame({'pred_label': label_collect, 'true_label': test_data['label']})
            f1 = f1_score(eval['true_label'], eval['pred_label'], average=average_method)
            if f1 > best_f1:
                best_f1 = f1
                best_model = file

            # Log stats
            loss_report = '  '.join(f"{key}: {np.mean(values):.4f}" for key, values in loss_collect.items())
            metrics_report = f"F1 Score: {f1:.4f}  Precision: {precision_score(eval['true_label'], eval['pred_label'], average=average_method):.4f}  Recall: {recall_score(eval['true_label'], eval['pred_label'], average=average_method):.4f}"
            print(f"{file} --> {metrics_report}  {loss_report}", file=log_file, flush=True)

    # Print best model
    print(f"Best model: {best_model} with F1 Score: {best_f1}", file=log_file)
    log_file.close()