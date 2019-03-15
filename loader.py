import re
import numpy as np
import pandas as pd
import spacy
import torch
import pathlib
from torchtext import data

MAX_CHARS = 20000
NLP = spacy.load('en')
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


class ToxicLoader():

    def __init__(self, root, tokenizer, vectors=None, fix_length=100, val_ratio=0.2, max_vocab=20000, min_freq=50, lower=False):
        self.val_ratio = val_ratio
        self.root = pathlib.Path(root)
        self.tokenizer = tokenizer
        self.vectors = vectors
        if not vectors is None:
            lower = True
        # create data field for the comments
        self.comment = data.Field(
            sequential=True,
            fix_length=fix_length,
            tokenize=self.tokenizer,
            pad_first=True,
            dtype=torch.long,
            lower=lower
        )
        self.get_dataset(max_vocab=max_vocab, min_freq=min_freq)


    def prepare_csv(self, data_folder='data', seed=999):
        pathlib.Path(self.root / 'cache').mkdir(parents=True, exist_ok=True)

        df_train = pd.read_csv(self.root / data_folder / 'train.csv')
        df_train["comment_text"] = df_train.comment_text.str.replace("\n", " ")
        idx = np.arange(df_train.shape[0])
        np.random.seed(seed)
        np.random.shuffle(idx)
        val_size = int(len(idx) * self.val_ratio)
        df_train.iloc[idx[val_size:], :].to_csv(
            self.root / 'cache' / 'dataset_train.csv', index=False)
        df_train.iloc[idx[:val_size], :].to_csv(
            self.root / 'cache' / 'dataset_val.csv', index=False)

        df_test = pd.merge(pd.read_csv(self.root / data_folder / 'test.csv'), pd.read_csv(self.root / data_folder / 'test_labels.csv'), on='id')
        df_test = df_test.drop(df_test[(df_test['toxic'] < 0)].index)
        df_test["comment_text"] = \
            df_test.comment_text.str.replace("\n", " ")
        df_test.to_csv(self.root / 'cache' / 'dataset_test.csv', index=False)
        
        print(' ' * 100, end='\r')


    def get_dataset(self, max_vocab=20000, min_freq=50):
        print('Preparing CSV files...', end='\r')
        self.prepare_csv()
        print(' ' * 100, end='\r')
        print('Reading the files...', end='\r')
        train, val, test = data.TabularDataset.splits(
            path='cache/', format='csv', skip_header=True,
            train='dataset_train.csv', validation='dataset_val.csv', test='dataset_test.csv',
            fields=[
                ('id', None),
                ('comment_text', self.comment),
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
        print(' ' * 100, end='\r')
        print('Building Vocabulary...', end='\r')
        self.comment.build_vocab(
            train, val, test,
            max_size=max_vocab,
            min_freq=min_freq,
            vectors=self.vectors
        )
        self.train = train
        self.val = val
        self.test = test

        print(' ' * 100, end='\r')
        print('Done preparing the datasets')


    @staticmethod
    def get_iterator(dataset, batch_size, train=False, shuffle=True, repeat=False):
        dataset_iter = data.Iterator(
            dataset, batch_size=batch_size, device=torch.device('cuda:0'),
            train=train, shuffle=shuffle, repeat=repeat,
            sort=False
        )
        return dataset_iter

    def train_iterator(self, batch_size):
        return BatchGenerator(self.get_iterator(self.train, batch_size, train=True))

    def test_iterator(self, batch_size):
        return BatchGenerator(self.get_iterator(self.test, batch_size))
    
    def val_iterator(self, batch_size):
        return BatchGenerator(self.get_iterator(self.val, batch_size))

    def vocab(self):
        return self.comment.vocab

    def vocab_size(self):
        return len(self.vocab())

    def emb_matrix(self):
        if self.vectors is None:
            return None
        else:
            return self.comment.vocab.vectors().cuda()