import re
import logging

import numpy as np
import pandas as pd
import spacy
import torch
from torchtext import data

NLP = spacy.load('en')
MAX_CHARS = 20000
VAL_RATIO = 0.2
LOGGER = logging.getLogger("toxic_dataset")


def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]


def prepare_csv(seed=999):
    df_train = pd.read_csv("data/train.csv")
    df_train["comment_text"] = df_train.comment_text.str.replace("\n", " ")
    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        "cache/dataset_val.csv", index=False)
    df_test = pd.read_csv("data/test.csv")
    df_test["comment_text"] = df_test.comment_text.str.replace("\n", " ")
    df_test.to_csv("cache/dataset_test.csv", index=False)


def get_dataset(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv()
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        dtype=torch.long,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('id', None),
            ('comment_text', comment),
            ('toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
            ('obscene', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
            ('threat', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
            ('insult', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False, dtype=torch.uint8)),
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment)
        ])
    LOGGER.debug("Building vocabulary...")
    comment.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train, val, test


def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=torch.device('cuda:0'),
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter

class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.target_fields = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
        self.data_field = 'comment_text'
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.data_field)
            Y = torch.transpose(torch.stack([getattr(batch, t) for t in self.target_fields]), 0, 1)
            yield (X, Y)