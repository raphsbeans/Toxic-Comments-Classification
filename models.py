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
    def __init__(self, vocab_size, emb_dim, hidden_size, vectors=None, freeze=True):
        super(CoolerNameNet, self).__init__()
        self.hidden_size = hidden_size
        if not vectors is None:
            self.embeddings = nn.Embedding.from_pretrained(vectors, freeze=freeze)
        else:
            self.embeddings = nn.Embedding(vocab_size, emb_dim)
        
        # may change to "spatial dropout" where we drop entire words
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(emb_dim, hidden_size, bidirectional=True)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, bidirectional=True)
        self.linear1 = nn.Linear(6 * hidden_size, 6)
        
    def init_hidden(self, batch_size):
        return torch.zeros((2, batch_size, self.hidden_size)).cuda() # (num_layers * num_directions, batch, hidden_size)
    
    def forward(self, data):
        # get batch size of input
        batch_size = data.shape[-1]
        
        # calculate the embeddings
        embedded = self.embeddings(data)
        
        # initiate hidden and cell state of lstm
        h_0 = self.init_hidden(batch_size)
        c_0 = self.init_hidden(batch_size)
        
        # pass the data through lstm and gru
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded, (h_0, c_0))
        gru_out, hidden_state = self.gru(lstm_out, hidden_state)
        
        # calculate max and avg values of the vectors associated with the words
        # need to adapt gru output shape for this
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1, 2, 0), 1).view(batch_size, -1)
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1, 2, 0), 1).view(batch_size, -1)
        
        # flatten hidden states of the 2 directions into one hidden state of 
        hidden_state = hidden_state.permute(1, 0, 2).contiguous().view(batch_size, -1)
                
        # concatenate all results and run linear layer to get output probs
        output = torch.cat([avg_pool, hidden_state, max_pool], dim=1)        
        output = self.linear1(output)
        
        return output.sigmoid()