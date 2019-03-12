import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time


class CoolNameNet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, batch_size, vectors=None, fine_tuning=True):
        super(CoolNameNet, self).__init__()
        self.batch_size = batch_size
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
    
    # change this ; need to implement metric function that run in gpu
    def evaluate(self, data_generator, loss_func, metric):
        with torch.no_grad():
            running_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for comments, targets in data_generator:
                    predictions = self.forward(comments)
                    loss = loss_func(predictions, targets.float())
                    running_loss += loss.item()
                    all_preds.append(predictions.detach().cpu().numpy())
                    all_targets.append(targets.detach().cpu().numpy())

                metric_value = metric(np.vstack(all_targets), np.vstack(all_preds))
                return running_loss / len(data_generator), metric_value

import time

def train(model, data_train, optimizer, loss_func, metric, data_val=None, batch_size=32, n_epochs=10, evaluate=True):
    def progress_bar(rate, total=30):
        bar = '=' * int(total * rate)
        if int(total * rate) < total:
            bar += '>'
            bar += '-' * (total - int(total * rate) - 1)

        return '[' + bar + ']'
    
    print('Starting Training...', end='\r')
    train_size = len(data_train)
    train_losses = []    
    if evaluate:
        val_avg_losses = []
        val_scores = []
        train_scores = []
    
    start_time = time.time()
    for epoch in range(n_epochs):        
        epoch_time = time.time()
        running_loss = 0

        if evaluate:
            all_preds = []
            all_targets = []

        counter = 0
        for inputs, targets in data_train:
            counter += 1
            if counter % 10 == 0:
                print('Epoch {} - status: {:.2f}% {}, elapsed time: {:.2f}s'.format(epoch, 
                                                                                    100 * counter/train_size,
                                                                                    progress_bar(counter/train_size),
                                                                                    time.time() - epoch_time), end='\r')
                #print('{} batches trained'.format(counter), end='\r')
                
            predictions = model(inputs)

            # Remember that torch accumulates gradient
            optimizer.zero_grad()

            loss = loss_func(predictions, targets.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if evaluate:
                all_preds.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        running_loss = running_loss / len(data_train)
        train_losses.append(running_loss)
        
        epoch_duration = time.time() - epoch_time
        avg_duration = (time.time() - start_time) / (epoch + 1)
        print('Epoch {} - avg_time/epoch: {:.2f}s - epoch_duration {:.2f}s'.format(epoch, avg_duration, epoch_duration))
        
        if evaluate:
            print('Evaluating the model on the validation set', end='\r')
            t_score = metric(np.vstack(all_targets), np.vstack(all_preds))
            train_scores.append(t_score)
            if not data_val is None:
                avg_loss, v_score = model.evaluate(data_val, loss_func, metric)
                val_avg_losses.append(avg_loss)
                val_scores.append(v_score)
            
            print('    Train Loss: {:.2f}, Train Score {:.2f}, Val. Loss: {:.2f}, Val. Score: {:.2f}'.format(running_loss, 
                                                                                                             t_score, 
                                                                                                             avg_loss,
                                                                                                             v_score))
        else:
            print('    Train Loss: {:.2f}'.format(running_loss))
    
    if evaluate:
        return train_losses, train_scores, val_avg_losses, val_scores
    else:
        return train_losses