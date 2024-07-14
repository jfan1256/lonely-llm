import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

import yaml
import wandb
import pandas as pd

from transformers import EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments

from utils.print import print_header
from utils.system import get_data, get_store_model, get_configs
from class_dataloader.dataloader_reddit_lonely import RedditLonelyMLM

if __name__ == '__main__':
    # Login
    config = yaml.load(open(get_configs() / 'api' / 'api.yaml', 'r'), Loader=yaml.Loader)
    wandb.login(key=config['wandb'])

    # Initialize Model
    print_header('Initialized Model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device='cuda')

    # Load data
    print_header('Load Data')
    data = pd.read_csv(get_data() / '0_lonely_posts_2019_2021.csv')
    data['body'] = data['body'].astype(str)
    data = data[data['body'].apply(lambda x: len(x.split()) >= 50)]
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=20050531)
    train_dataset = RedditLonelyMLM(data=train_data, tokenizer=tokenizer)
    validation_dataset = RedditLonelyMLM(data=val_data, tokenizer=tokenizer)

    # Finetune
    print_header('Finetune BERT')
    training_args = TrainingArguments(
        output_dir=get_store_model() / 'bert_lonely_finetune',
        num_train_epochs=1000,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=None,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Start training
    trainer.train()