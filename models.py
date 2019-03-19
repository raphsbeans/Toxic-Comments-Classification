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
    
class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        
        if not config['vectors'] is None:
            self.encoder = nn.Embedding.from_pretrained(config['vectors'])
        else:
            self.encoder = nn.Embedding(config['ntoken'], config['emb_dim'])
            
        if not config['fine_tuning']:
            self.encoder.weight.require_grad = False
        
        
        #self.encoder = nn.Embedding(config['ntoken'], config['emb_dim'])
        self.bilstm = nn.LSTM(config['emb_dim'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        
    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        emb = self.drop(self.encoder(inp))
        outp = self.bilstm(emb, hidden)[0]
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden( self, bsz ):
        # trick see https://discuss.pytorch.org/t/when-to-initialize-lstm-hidden-state/2323
        weight = next(self.parameters())
        # attention: hidden contains both hidden state and cell state
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid))
    
class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == 1).float()) #pad == 1
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.sigmoid = nn.Sigmoid()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, inp, hidden):
        outp, attention = self.encoder.forward(inp, hidden)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        pred = self.sigmoid(pred)
        out_bool = True
        if type(self.encoder) == BiLSTM:
            attention = None
            out_bool = False
        return pred, attention, out_bool 

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]
    