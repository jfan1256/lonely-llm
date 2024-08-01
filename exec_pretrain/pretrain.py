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
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling

from utils.system import get_configs
from utils.print import print_header
from class_models.model_utils import set_seed
from class_models.plot import plot_diagnostics
from class_dataloader.dataloader import Pretrain
from class_dataloader.utils import create_dataloader, create_sampler
from class_models.train_utils import init_distributed_mode, get_rank, get_world_size, is_main_process
from class_models.train_utils import MetricLogger, SmoothedValue, warmup_lr_schedule, step_lr_schedule

# Training Loop
def train(epoch, model, dataloader, optimizer, configs):
    # Set model to train
    model.train()

    # Set dataloader to a new random sample
    if configs['num_gpu'] > 1:
        dataloader.sampler.set_epoch(epoch)

    # Initialize MetricLogger
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss_bert', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.8f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 10

    # Train epoch
    for step, (batch) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        # Warmup learning rate for first epoch
        if epoch == 1:
            warmup_lr_schedule(optimizer, step, configs['warmup_steps'], configs['warmup_lr'], configs['init_lr'])

        # Get input and label
        inputs = {key: value.to(configs['device']) for key, value in batch.items() if key != "labels"}
        labels = batch['labels'].to(configs['device'])

        # Reset gradients
        optimizer.zero_grad()

        # Train model
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward propagation
        loss.backward()
        optimizer.step()

        # Update metric logger
        metric_logger.update(loss_bert=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.8f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

# Evaluation Loop
def eval(model, dataloader, configs):
    # Set model to eval
    model.eval()

    # Create loss collectors
    accumulators = {'loss_bert': []}

    # Eval
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Get input and label
            inputs = {key: value.to(configs['device']) for key, value in batch.items() if key != 'labels'}
            labels = batch['labels'].to(configs['device'])

            # Train model
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Accumulate losses
            accumulators['loss_bert'].append(loss.item())

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
    tokenizer = BertTokenizer.from_pretrained(configs['bert_model'])
    model = BertForMaskedLM.from_pretrained(configs['bert_model'])
    model.to(configs['device'])

    # Initialize dataloader
    print_header("Load Data")
    # Load data
    # data = pd.read_csv(configs['pretrain_path'])
    all_files = [os.path.join(configs['pretrain_path'], file) for file in os.listdir(configs['pretrain_path']) if file.endswith('.csv')]
    data = pd.concat((pd.read_csv(file) for file in tqdm(all_files, desc='Load Data')), ignore_index=True)

    # Shuffle data
    data = data.sample(frac=1, random_state=20050531).reset_index(drop=True)

    # Split data 80/20 train/val
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=20050531)

    # Create dataset
    train_dataset = Pretrain(train_data, tokenizer)
    val_dataset = Pretrain(val_data, tokenizer)

    # Initialize dataloader
    print_header("Initialize Dataloader")

    # MLM Dynamic Masking
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Data setup
    if configs['num_gpu'] > 1:
        # Create sampler
        num_tasks = get_world_size()
        global_rank = get_rank()

        # Create samplers for Distributed Data Parallelism
        train_sampler = create_sampler(datasets=[train_dataset], shuffles=[True], num_tasks=num_tasks, global_rank=global_rank)

        # Create dataloaders
        train_dataloader = create_dataloader(datasets=[train_dataset], samplers=train_sampler, batch_size=[configs['batch_size']], num_workers=[4], is_trains=[True], collate_fns=[data_collator])[0]
        val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=False, collate_fn=data_collator)
    else:
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=True, drop_last=True, collate_fn=data_collator)
        val_dataloader = DataLoader(val_dataset, batch_size=configs['batch_size'], num_workers=4, shuffle=False, collate_fn=data_collator)

    # Initialize optimizer
    print_header("Initialize Optimizer")
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=configs['init_lr'], weight_decay=configs['weight_decay'])

    # Start epoch
    start_epoch = 0

    # Load checkpoint model
    if configs['train_checkpoint'] != '':
        print_header("Load Checkpoint")
        checkpoint = torch.load(configs['train_checkpoint'], map_location='cpu')
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
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
        step_lr_schedule(optimizer, epoch - 1, configs['init_lr'], configs['min_lr'], configs['lr_decay_rate'])

        # Train model
        train_stats = train(epoch, model, train_dataloader, optimizer, configs)

        # Synchronize multi-gpu
        if configs['num_gpu'] > 1:
            dist.barrier()
            torch.cuda.empty_cache()

        # Check main process (rank == 0)
        if is_main_process():
            # Save model
            save_obj = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
            torch.save(save_obj, os.path.join(configs['output_dir'], 'checkpoint_%02d.pth' % epoch))

            # # Eval model
            # val_stats = eval(model_without_ddp, val_dataloader, configs)
            # eval_loss = sum(val_stats.values())
            #
            # # Collect losses
            # for key in val_stats.keys():
            #     if key not in loss_collect['train']:
            #         loss_collect['train'][key] = []
            #     if key not in loss_collect['val']:
            #         loss_collect['val'][key] = []
            #     loss_collect['train'][key].append(train_stats[key])
            #     loss_collect['val'][key].append(val_stats[key])
            #
            # # Log eval loss
            # loss_details = '  '.join([f"{name}: {value:.8f}" for name, value in val_stats.items()])
            # print(f"{loss_details}")
            #
            # # Check for improvement and update best_checkpoint.pth
            # if eval_loss < best_loss:
            #     best_loss = eval_loss
            #     epochs_without_improvement = 0
            #     save_obj = {'model': model_without_ddp.state_dict(), 'optimizer': optimizer.state_dict(), 'config': configs, 'epoch': epoch}
            #     torch.save(save_obj, os.path.join(configs['output_dir'], 'best_checkpoint.pth'))
            # else:
            #     epochs_without_improvement += 1
            #
            # # Early stopping check
            # if epochs_without_improvement >= patience:
            #     print_header(f"Early stop at {epoch}")
            #     break
            #
            # # Log stats
            # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, **{f'val_{k}': "{:.8f}".format(v) for k, v in val_stats.items()}, 'epoch': epoch}
            # with open(os.path.join(configs['output_dir'], "log.txt"), "a") as f:
            #     f.write(json.dumps(log_stats) + "\n")

        # Synchronize multi-gpu
        if configs['num_gpu'] > 1:
            dist.barrier()
            torch.cuda.empty_cache()

    # Calculate total time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return loss_collect

if __name__ == "__main__":
    # Set seed
    set_seed(20050531)

    # Use most efficient algorithm
    cudnn.benchmark = True

    # Get configs
    configs = yaml.load(open(get_configs() / 'pretrain' / 'bert_lonely.yaml', 'r'), Loader=yaml.Loader)

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