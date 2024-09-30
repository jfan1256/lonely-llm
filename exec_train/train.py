import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import time
import yaml
import torch
import argparse
import datetime
import numpy as np
import pandas as pd
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from utils.print import print_header
from utils.system import get_configs
from class_models.model_utils import set_seed
from class_dataloader.dataloader import Train
from class_models.plot import plot_diagnostics
from class_models.psychspt import init_psychspt
from class_dataloader.utils import create_dataloader, create_sampler
from class_models.train_utils import MetricLogger, SmoothedValue, cosine_lr_schedule
from class_models.train_utils import init_distributed_mode, get_rank, get_world_size, is_main_process, freeze_param

# Training Loop
def train(epoch, model, dataloader, optimizer, configs):
    # Set model to train
    model.train()

    # Set dataloader to a new random sample
    if configs['num_gpu'] > 1:
        dataloader.sampler.set_epoch(epoch)

    # Initialize MetricLogger
    metric_logger = MetricLogger(delimiter="  ")
    loss_keys = ['loss_ce', 'loss_focal', 'loss_dice', 'loss_tversky', 'loss_center', 'loss_angular', 'loss_contrast', 'loss_reason', 'loss_perplex', 'loss_embed_match']
    for key in loss_keys:
        metric_logger.add_meter(key, SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('loss_total', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('lr_bert', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    if configs['decoder_only'] != 'yes':
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
        for key, value in losses.items():
            metric_logger.update(**{key: value.item()})
        if configs['decoder_only'] == 'yes':
            metric_logger.update(loss_total=loss, lr_bert=optimizer.param_groups[0]["lr"])
        else:
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
    loss_keys = ['loss_ce', 'loss_focal', 'loss_dice', 'loss_tversky', 'loss_center', 'loss_angular', 'loss_contrast', 'loss_reason', 'loss_perplex', 'loss_embed_match']
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
def main(args, configs):
    # Initialize multi-gpu distributed process
    if configs['num_gpu'] > 1:
        print_header("Initialize Distributed Mode")
        init_distributed_mode(args)

    # Initialize model
    print_header("Initialize Model")
    bert_reddit_model = init_psychspt(pretrained=None, configs=configs)
    model = bert_reddit_model.to(device=configs['train_device'])

    # Initialize dataloader
    print_header("Initialize Dataloader")
    # Load data
    train_data = pd.read_csv(configs['train_path'])
    train_data = train_data.fillna("no reason")
    val_data = pd.read_csv(configs['val_path'])
    val_data = val_data.fillna("no reason")

    # Shuffle data
    train_data = train_data.sample(frac=1, random_state=20050531).reset_index(drop=True)

    # Create dataset
    train_dataset = Train(data=train_data)
    val_dataset = Train(data=val_data)

    # Data setup
    if configs['num_gpu'] > 1:
        # Create sampler
        num_tasks = get_world_size()
        global_rank = get_rank()

        # Create samplers for Distributed Data Parallelism
        train_sampler = create_sampler(datasets=[train_dataset], shuffles=[True], num_tasks=num_tasks, global_rank=global_rank)
        val_sampler = create_sampler(datasets=[val_dataset], shuffles=[False], num_tasks=num_tasks, global_rank=global_rank)

        # Create dataloaders
        train_dataloader = create_dataloader(datasets=[train_dataset], samplers=train_sampler, batch_size=[configs['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[None])[0]
        val_dataloader = create_dataloader(datasets=[val_dataset], samplers=val_sampler, batch_size=[configs['batch_size']], num_workers=[4], is_trains=[False], collate_fns=[None])[0]
    else:
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=False, drop_last=False)

    # Initialize optimizer
    print_header("Initialize Optimizer")
    encoder_decoder_params = []

    # Handle freezing layers if training only decoder
    if configs['decoder_only'] == 'yes':
        freeze_param(model.text_encoder)
        freeze_param(model.mlp_task)
        freeze_param(model.mlp_sentiment)
        encoder_decoder_params += list(model.text_decoder.parameters())
        # Initialize optimizer
        optimizer = torch.optim.AdamW([
            {'params': encoder_decoder_params, 'lr': configs['bert_lr']},
        ], weight_decay=configs['weight_decay'])
    else:
        # Train decoder parameters if decoder losses are implemented
        if 'loss_reason' in configs['loss'] or 'loss_perplex' in configs['loss'] or 'loss_embed_match' in configs['loss']:
            encoder_decoder_params += list(model.text_encoder.parameters()) + list(model.text_decoder.parameters())
        else:
            encoder_decoder_params += list(model.text_encoder.parameters())

        # MLP parameters
        mlp_params = list(model.mlp_task.parameters()) + list(model.mlp_sentiment.parameters())

        # Initialize optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': encoder_decoder_params, 'lr': configs['bert_lr']},
            {'params': mlp_params, 'lr': configs['mlp_lr']}
        ], weight_decay=configs['weight_decay'])

    # Start epoch
    start_epoch = 0

    # Load checkpoint model
    if configs['train_checkpoint'] != '':
        print_header("Load Checkpoint")
        checkpoint = torch.load(configs['train_checkpoint'], map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)

        # Skip this step if training only decoder from classifier checkpoint
        if configs['decoder_only'] != 'yes':
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']

    # Store model without DDP for saving
    model_without_ddp = model
    if configs['num_gpu'] > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

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
        cosine_lr_schedule(optimizer, 0, epoch - 1, configs['max_epoch'], configs['bert_lr'], configs['min_lr'])
        if configs['decoder_only'] != 'yes':
            cosine_lr_schedule(optimizer, 1, epoch - 1, configs['max_epoch'], configs['mlp_lr'], configs['min_lr'])

        # Train model
        train_stats = train(epoch, model, train_dataloader, optimizer, configs)

        # Check main process (rank == 0)
        if is_main_process():
            # Eval model
            val_stats = eval(model_without_ddp, val_dataloader, configs)
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
                save_obj = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
                torch.save(save_obj, os.path.join(configs['output_dir'], 'best_checkpoint.pth'))
            else:
                epochs_without_improvement += 1

            # Early stopping check
            if epochs_without_improvement >= patience:
                print_header(f"Early stop at {epoch}")
                break

            # Save model and log results
            save_obj = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
            torch.save(save_obj, os.path.join(configs['output_dir'], 'checkpoint_%02d.pth' % epoch))
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': "{:.8f}".format(v) for k, v in val_stats.items()}, 'epoch': epoch}
            with open(os.path.join(configs['output_dir'], "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Synchronize multi-gpu
        if configs['num_gpu'] > 1:
            dist.barrier()
            torch.cuda.empty_cache()

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return loss_collect

if __name__ == '__main__':
    # Set seed
    set_seed(20050531)

    # Use most efficient algorithm
    cudnn.benchmark = True

    # Get configs
    configs = yaml.load(open(get_configs() / 'train' / 'psychspt.yaml', 'r'), Loader=yaml.Loader)

    # Parse configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', help='Number of distributed processes (analogous to number of gpus)')
    parser.add_argument('--dist_url', help='URL used to set up distributed training')
    parser.add_argument('--distributed', help='Distributed or not')
    parser.add_argument('--package', help='Distributed package name')
    args = parser.parse_args()

    # Set package
    args.package = configs['package']
    if args.package == 'gloo':
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'

    # Create output directory and save configs
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True)
    yaml.dump(configs, open(os.path.join(configs['output_dir'], 'configs.yaml'), 'w'))

    # Execute main
    loss_data = main(args, configs)

    # Plot curves
    plot_diagnostics(loss_data, configs['output_dir'])