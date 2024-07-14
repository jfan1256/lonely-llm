import warnings
warnings.filterwarnings('ignore')

import os
import json
import time
import yaml
import torch
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from utils.print import print_header
from utils.system import get_configs
from class_models.utils import set_seed
from exec_train.plot import plot_diagnostics
from class_models.bert_lonely import init_bert_lonely
from class_dataloader.dataloader_reddit_lonely import RedditLonelyTrain
from class_models.metrics import MetricLogger, SmoothedValue, cosine_lr_schedule, warmup_lr_schedule, step_lr_schedule


# Training Loop
def train(epoch, model, dataloader, optimizer, configs):
    # Loss collector
    binary_loss_collector = []
    sentiment_loss_collector = []
    constrat_loss_collector = []
    reason_loss_collector = []

    # Set model to train
    model.train()

    # Metric Loggers
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_total', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_binary', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_sentiment', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_constrast', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_reason', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.8f}'))

    # Print frequency
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Iterate through images
    for i, (index, prompt, label, reason) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # Warmup learning rate for first epoch
        if epoch == 1:
            warmup_lr_schedule(optimizer, i, configs['warmup_steps'], configs['warmup_lr'], configs['init_lr'])

        # Set device
        label = torch.tensor(label, dtype=torch.float).to(configs['train_device'])

        # Get sentiment
        sentiment = model.get_sentiment(prompt)
        sentiment = torch.tensor(sentiment, dtype=torch.float).to(configs['train_device'])

        # Zero out gradients
        optimizer.zero_grad()

        # Train model and calculate loss
        loss_binary, loss_sentiment, loss_constrast, loss_reason, prob = model(index=index, prompt=prompt, label=label, reason=reason, sentiment=sentiment, device=configs['train_device'])
        binary_loss_collector.append(loss_binary.item())
        sentiment_loss_collector.append(loss_sentiment.item())
        constrat_loss_collector.append(loss_constrast.item())
        reason_loss_collector.append(loss_reason.item())
        loss = configs['alpha'] * loss_binary + (1 - configs['alpha']) * loss_sentiment + loss_constrast + loss_reason

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update metric loggers
        metric_logger.update(loss_total=loss.item())
        metric_logger.update(loss_binary=loss_binary.item())
        metric_logger.update(loss_sentiment=loss_sentiment.item())
        metric_logger.update(loss_constrast=loss_constrast.item())
        metric_logger.update(loss_reason=loss_reason.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}, binary_loss_collector, sentiment_loss_collector, constrat_loss_collector, reason_loss_collector

# Evaluation loop
def eval(model, dataloader, configs):
    model.eval()
    binary_loss_collector = []
    sentiment_loss_collector = []
    constrast_loss_collector = []
    reason_loss_collector = []

    # Do not compute gradients
    with torch.no_grad():
        for (index, prompt, label, reason) in tqdm(dataloader, desc="Validating"):
            # Set device
            label = torch.tensor(label, dtype=torch.float).to(configs['train_device'])

            # Get sentiment
            sentiment = model.get_sentiment(prompt)
            sentiment = torch.tensor(sentiment, dtype=torch.float).to(configs['train_device'])

            # Evaluate model and calculate loss
            loss_binary, loss_sentiment, loss_constrast, loss_reason, prob = model(index=index, prompt=prompt, label=label, reason=reason, sentiment=sentiment, device=configs['train_device'])
            binary_loss_collector.append(loss_binary.item())
            sentiment_loss_collector.append(loss_sentiment.item())
            constrast_loss_collector.append(loss_constrast.item())
            reason_loss_collector.append(loss_reason.item())

    # Average validation loss across dataloader
    binary_loss = np.mean(binary_loss_collector)
    sentiment_loss = np.mean(sentiment_loss_collector)
    constrast_loss = np.mean(constrast_loss_collector)
    reason_loss = np.mean(reason_loss_collector)
    return binary_loss, sentiment_loss, constrast_loss, reason_loss

# Main
def main(configs):
    # Initialize model
    print_header("Initialize Model")
    bert_reddit_model = init_bert_lonely(pretrained=None, configs=configs)
    model = bert_reddit_model.to(device=configs['train_device'])

    # Initialize dataloader
    print_header("Initialize Dataloader")
    train_data = pd.read_csv(configs['train_path'])
    train_data = train_data.sample(frac=1, random_state=20050531).reset_index(drop=True)
    val_data = pd.read_csv(configs['val_path'])
    train_dataset = RedditLonelyTrain(data=train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, drop_last=True)
    val_dataset = RedditLonelyTrain(data=val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, drop_last=False)

    # Initialize optimizer
    print_header("Initialize Optimizer")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=configs['init_lr'], weight_decay=configs['weight_decay'])

    # Load checkpoint
    start_epoch = 0

    # Early stop params
    patience = configs['early_stop']
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Loss collectors
    binary_loss_train_collect, binary_loss_val_collect = [[]], [[]]
    sentiment_loss_train_collect, sentiment_loss_val_collect = [[]], [[]]
    constrast_loss_train_collect, constrast_loss_val_collect = [[]], [[]]
    reason_loss_train_collect, reason_loss_val_collect = [[]], [[]]

    # Start training
    print_header("Start Training")
    start_time = time.time()
    for epoch in range(start_epoch + 1, configs['max_epoch'] + 1):
        # Print
        print_header(f"Epoch {epoch}")

        # # Step the learning rate
        # cosine_lr_schedule(optimizer, epoch, configs['max_epoch'], configs['init_lr'], configs['min_lr'])
        step_lr_schedule(optimizer, epoch - 1, configs['init_lr'], configs['min_lr'], configs['lr_decay_rate'])

        # Train model
        train_stats, binary_loss_train_collector, sentiment_loss_train_collector, constrast_loss_train_collector, reason_loss_train_collector = train(epoch=epoch, model=model, dataloader=train_dataloader, optimizer=optimizer, configs=configs)
        binary_loss_train_collect[0].extend(binary_loss_train_collector)
        sentiment_loss_train_collect[0].extend(sentiment_loss_train_collector)
        constrast_loss_train_collect[0].extend(constrast_loss_train_collector)
        reason_loss_train_collect[0].extend(reason_loss_train_collector)

        # Evaluate model
        binary_loss_val, sentiment_loss_val, constrast_loss_val, reason_loss_val = eval(model=model, dataloader=val_dataloader, configs=configs)
        eval_loss = configs['alpha'] * binary_loss_val + (1 - configs['alpha']) * sentiment_loss_val + constrast_loss_val + reason_loss_val
        print(f"loss_eval: {eval_loss:.4f}  loss_binary: {binary_loss_val:.4f}  loss_sentiment: {sentiment_loss_val:.4f}  loss_constrast: {constrast_loss_val:.4f}  loss_reason: {reason_loss_val:.4f}")

        binary_loss_val_collect[0].append(binary_loss_val)
        sentiment_loss_val_collect[0].append(sentiment_loss_val)
        constrast_loss_val_collect[0].append(constrast_loss_val)
        reason_loss_val_collect[0].append(reason_loss_val)

        # Check for improvement
        if eval_loss < best_loss:
            best_loss = eval_loss
            epochs_without_improvement = 0
            # Save the model as the best
            save_obj = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
            torch.save(save_obj, os.path.join(configs['output_dir'], 'best_checkpoint.pth'))
        else:
            epochs_without_improvement += 1

        # Early stopping check
        if epochs_without_improvement >= patience:
            print_header(f"Early stop at {epoch}")
            break

        # Save model and log results
        save_obj = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
        torch.save(save_obj, os.path.join(configs['output_dir'], 'checkpoint_%02d.pth' % epoch))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, 'eval_loss': eval_loss}
        with open(os.path.join(configs['output_dir'], "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return binary_loss_train_collect, binary_loss_val_collect, sentiment_loss_train_collect, sentiment_loss_val_collect, constrast_loss_train_collect, constrast_loss_val_collect, reason_loss_train_collect, reason_loss_val_collect

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Get configs
    configs = yaml.load(open(get_configs() / 'train' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)

    # Create output directory and save configs
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True)
    yaml.dump(configs, open(os.path.join(configs['output_dir'], 'configs.yaml'), 'w'))

    # Execute main
    binary_loss_train_collect, binary_loss_val_collect, sentiment_loss_train_collect, sentiment_loss_val_collect, constrast_loss_train_collect, constrast_loss_val_collect, reason_loss_train_collect, reason_loss_val_collect = main(configs)

    # Plot curves
    plot_diagnostics(binary_loss_train_collect, binary_loss_val_collect, sentiment_loss_train_collect, sentiment_loss_val_collect, constrast_loss_train_collect, constrast_loss_val_collect, reason_loss_train_collect, reason_loss_val_collect)