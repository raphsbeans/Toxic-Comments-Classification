"""Improved dataset loader for Toxic Comment dataset from Kaggle
Tested against:
* Python 3.6
* Numpy 1.14.0
* Pandas 0.22.0
* PyTorch 0.4.0a0+f83ca63 (should be very close to 0.3.0)
* torchtext 0.2.1
* spacy 2.0.5
* joblib 0.11
"""
import re
import logging

import numpy as np
import pandas as pd
import spacy
import torch
from joblib import Memory
from torchtext import data
from sklearn.model_selection import KFold

NLP = spacy.load('en')
MAX_CHARS = 20000
LOGGER = logging.getLogger("toxic_dataset")
MEMORY = Memory(cachedir="cache/", verbose=1)


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


def prepare_csv():
    df_train = pd.read_csv("data/train.csv")
    df_train["comment_text"] = df_train.comment_text.str.replace("\n", " ")
    df_train.to_csv("cache/dataset_train.csv", index=False)
    df_test = pd.read_csv("data/test.csv")
    df_test["comment_text"] = df_test.comment_text.str.replace("\n", " ")
    df_test.to_csv("cache/dataset_test.csv", index=False)


@MEMORY.cache
def read_files(fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only support all lower case
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv()
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        dtype=torch.cuda.LongTensor,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train = data.TabularDataset(
        path='cache/dataset_train.csv', format='csv', skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment),
            ('toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
            ('obscene', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
            ('threat', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
            ('insult', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
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
        train, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train.examples, test.examples, comment


def get_dataset(fix_length=100, lower=False, vectors=None, n_folds=5, seed=999):
    train_exs, test_exs, comment = read_files(
        fix_length=fix_length, lower=lower, vectors=vectors)

    kf = KFold(n_splits=n_folds, random_state=seed)

    fields = [
        ('id', None),
        ('comment_text', comment),
        ('toxic', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
        ('severe_toxic', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
        ('obscene', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
        ('threat', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
        ('insult', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
        ('identity_hate', data.Field(
            use_vocab=False, sequential=False, dtype=torch.cuda.ByteTensor)),
    ]

    def iter_folds():
        train_exs_arr = np.array(train_exs)
        for train_idx, val_idx in kf.split(train_exs_arr):
            yield (
                data.Dataset(train_exs_arr[train_idx], fields),
                data.Dataset(train_exs_arr[val_idx], fields),
            )

    test = data.Dataset(test_exs, fields[:2])
    return iter_folds(), test


def get_iterator(dataset, batch_size, train=True, shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter