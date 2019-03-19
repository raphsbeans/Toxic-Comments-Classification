import re
import numpy as np
import pandas as pd
import spacy
import torch
import pathlib
from torchtext import data

MAX_CHARS = 20000
NLP = spacy.load('en')
def my_tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [x.text for x in NLP.tokenizer(comment) if x.text != " "]


def comment_to_tensor(comment, tokenizer, vocab, cuda=True, fix_length=None):
    tokens = tokenizer(comment)
    if fix is None:
        fix_length = len(tokens)
        
    comment_tensor = torch.zeros(fix_length, dtype=torch.long)
    i = 0
    for t in tokens:
        i += 1
        if i > fix_length:
            break
        
        comment_tensor[i] = vocab.stoi[t.lower()]
    
    if i < fix_length:
        for j in range(i, fix_length):
            comment_tensor[i] = vocab.stoi['<pad>']
    
    comment_tensor = comment_tensor.view(-1, 1)
    if cuda:
        comment_tensor = comment_tensor.cuda()
        
    return comment_tensor


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

    def __init__(self, tokenizer=my_tokenizer, root='.', vectors=None, fix_length=100, val_ratio=0.2, max_vocab=20000, min_freq=50, lower=False):
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
            return self.comment.vocab.vectors.cuda()
        
    def generate_submission(self, model, filename='submission.csv'):
        data_fields = [('id', None), ('comment_text', self.comment)]

        test_dataset = data.TabularDataset(path = self.root/'data'/'test.csv', format='csv', fields=data_fields, skip_header=True)
        test_itr = data.Iterator(test_dataset, batch_size=32, 
                                 device=torch.device('cuda:0'), 
                                 sort=False, sort_within_batch=False, 
                                 repeat=False, shuffle=False)

        df_test = pd.read_csv(self.root / 'data' / 'test.csv')

        predictions = []
        with torch.no_grad():
            for batch in test_itr:
                inputs = getattr(batch, 'comment_text')
                probs = model(inputs)
                predictions.append(probs.cpu())
                
        all_predictions = np.vstack(predictions)
        for i, col in enumerate(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]):
            df_test[col] = all_predictions[:, i]
        
        pathlib.Path(self.root / 'cache').mkdir(parents=True, exist_ok=True)
        df_test.drop('comment_text', axis=1).to_csv(self.root / 'cache' / filename, index=False)