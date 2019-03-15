import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CoolNameNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, vectors=None, fine_tuning=True):
        super(CoolNameNet, self).__init__()
        self.hidden_size = hidden_size
        
        if not vectors is None:
            self.embeddings = nn.Embedding.from_pretrained(vectors)
        else:
            self.embeddings = nn.Embedding(vocab_size, emb_dim)
            
        if not fine_tuning:
            self.embeddings.weight.require_grad = False
        
        # just some basic rnn to test the data_iterators
        self.lstm = nn.LSTM(emb_dim, hidden_size, num_layers=2, bidirectional=True, dropout=0.1)
        
        self.linear1 = nn.Linear(emb_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 6)
        
    def init_hidden(self, batch_size):
        return (torch.zeros((4, batch_size, self.hidden_size)).cuda(), 
                torch.zeros((4, batch_size, self.hidden_size)).cuda()) # (num_layers * num_directions, batch, hidden_size)
    
    def forward(self, data):
        self.hidden_state = self.init_hidden(data.shape[1])
        
        embedded = self.embeddings(data)
        lstm_out, self.hidden_state = self.lstm(embedded, self.hidden_state)
        output = self.linear1(lstm_out[-1])
        output = F.relu(output)
        output = self.linear2(output)
        
        probs = output.sigmoid()
        return probs


class CoolerNameNet(nn.Module):

    def __init__(self, vocab_size, emb_dim, vectors1=None, vectors2=None):
        super(CoolerNameNet, self).__init__()


    def forward(self):
        pass

    def init_hidden(self, batch_size):
        pass