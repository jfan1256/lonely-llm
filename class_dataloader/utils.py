import re
import jit
import torch
import unicodedata
import numpy as np
from torch.utils.data import DataLoader

# Get word count of dataset column
@jit
def fast_word_count(texts):
    counts = np.zeros(len(texts), dtype=np.int32)
    for i, text in enumerate(texts):
        counts[i] = len(text.split())
    return counts

# Preprocess text
def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = re.sub(r'\s+', ' ', text).strip()
    contractions = {
        "can't": "cannot", "didn't": "did not", "don't": "do not",
        "it's": "it is", "i'm": "i am", "you're": "you are",
        "he's": "he is", "she's": "she is", "that's": "that is",
        "there's": "there is", "what's": "what is", "who's": "who is"
    }
    words = text.split()
    reformed = [contractions[word] if word in contractions else word for word in words]
    text = " ".join(reformed)
    return text

# Create DDP sampler
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers

# Create dataloader
def create_dataloader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders