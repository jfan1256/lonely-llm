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
from class_dataloader.dataloader import Train
from class_models.metrics import MetricLogger, SmoothedValue, cosine_lr_schedule

# Update metrics
def update_metrics(metric_logger, losses, update_global=True):
    for key, value in losses.items():
        if update_global:
            metric_logger.update(**{key: value.item()})
# Training Loop
def train(epoch, model, dataloader, optimizer, configs):
    # Set model to train
    model.train()

    # Initialize MetricLogger
    metric_logger = MetricLogger(delimiter="  ")
    loss_keys = ['loss_focal', 'loss_dice', 'loss_tversky', 'loss_center', 'loss_angular', 'loss_contrast', 'loss_reason', 'loss_perplex', 'loss_embed_match']
    for key in loss_keys:
        metric_logger.add_meter(key, SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_total', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('lr_bert', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('lr_mlp', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    # Train epoch
    for i, (index, narrative, label, reason) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # Get label and sentiment
        label = torch.tensor(label, dtype=torch.float).to(configs['train_device'])
        sentiment = model.get_sentiment(narrative)
        sentiment = torch.tensor(sentiment, dtype=torch.float).to(configs['train_device'])

        # Reset gradients
        optimizer.zero_grad()

        # Train model
        losses = model(index=index, narrative=narrative, label=label, reason=reason, sentiment=sentiment, device=configs['train_device'])

        # Total loss
        loss = sum(loss for loss in losses.values())

        # Backward propagation
        loss.backward()
        optimizer.step()

        # Update metrics
        update_metrics(metric_logger, losses)
        metric_logger.update(loss_total=loss, lr_bert=optimizer.param_groups[0]["lr"], lr_mlp=optimizer.param_groups[1]["lr"])

    # Return stats
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

# Evaluation loop
def eval(model, dataloader, configs):
    # Set model to eval
    model.eval()

    # Create loss collectors
    loss_keys = ['loss_focal', 'loss_dice', 'loss_tversky', 'loss_center', 'loss_angular', 'loss_contrast', 'loss_reason', 'loss_perplex', 'loss_embed_match']
    accumulators = {key: [] for key in loss_keys}

    # Eval
    with torch.no_grad():
        for (index, narrative, label, reason) in tqdm(dataloader, desc="Validating"):
            # Get label and sentiment
            label = torch.tensor(label, dtype=torch.float).to(configs['train_device'])
            sentiment = model.get_sentiment(narrative)
            sentiment = torch.tensor(sentiment, dtype=torch.float).to(configs['train_device'])

            # Evaluate model
            losses = model(index=index, narrative=narrative, label=label, reason=reason, sentiment=sentiment, device=configs['train_device'])

            # Accumulate losses
            for key in losses:
                accumulators[key].append(losses[key].item())

    # Calculate average
    average_losses = {key: np.mean(values) for key, values in accumulators.items() if values}
    return average_losses

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
    train_dataset = Train(data=train_data)
    train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, drop_last=True)
    val_dataset = Train(data=val_data)
    val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False, drop_last=False)

    # Initialize optimizer
    print_header("Initialize Optimizer")
    encoder_decoder_params = list(model.text_encoder.parameters()) + list(model.text_decoder.parameters())
    mlp_params = list(model.mlp_lonely.parameters()) + list(model.mlp_sentiment.parameters())
    optimizer = torch.optim.AdamW([
        {'params': encoder_decoder_params, 'lr': configs['bert_lr']},
        {'params': mlp_params, 'lr': configs['mlp_lr']}
    ], weight_decay=configs['weight_decay'])

    # Start epoch
    start_epoch = 0

    # Early stop params
    patience = configs['early_stop']
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Loss collectors
    loss_collect = {'train': {}, 'val': {}}

    # Start training
    print_header("Start Training")
    start_time = time.time()
    for epoch in range(start_epoch + 1, configs['max_epoch'] + 1):
        # Print
        print_header(f"Epoch {epoch}")

        # Step the learning rate
        cosine_lr_schedule(optimizer, 0, epoch, configs['max_epoch'], configs['bert_lr'], configs['min_lr'])
        cosine_lr_schedule(optimizer, 1, epoch, configs['max_epoch'], configs['mlp_lr'], configs['min_lr'])

        # Train model
        train_stats = train(epoch, model, train_dataloader, optimizer, configs)

        # Eval model
        val_stats = eval(model, val_dataloader, configs)
        eval_loss = sum(val_stats.values())
        val_stats['loss_total'] = eval_loss

        # Collect losses
        for key in val_stats.keys():
            if key not in loss_collect['train']:
                loss_collect['train'][key] = []
            if key not in loss_collect['val']:
                loss_collect['val'][key] = []
            loss_collect['train'][key].append(train_stats[key])
            loss_collect['val'][key].append(val_stats[key])

        # Log eval loss
        loss_details = '  '.join([f"{name}: {value:.8f}" for name, value in val_stats.items()])
        print(f"{loss_details}")

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
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': "{:.8f}".format(v) for k, v in val_stats.items()}, 'epoch': epoch}
        with open(os.path.join(configs['output_dir'], "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return loss_collect

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Get configs
    configs = yaml.load(open(get_configs() / 'train' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)

    # Create output directory and save configs
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True)
    yaml.dump(configs, open(os.path.join(configs['output_dir'], 'configs.yaml'), 'w'))

    # Execute main
    loss_data = main(configs)

    # Plot curves
    plot_diagnostics(loss_data, configs['output_dir'])