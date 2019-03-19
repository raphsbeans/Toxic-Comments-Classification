import torch
import time
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import utils

def evaluate(model, data_generator, loss_func, metric, has_hidden = False):
    with torch.no_grad():
        running_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for comments, targets in data_generator:
                if has_hidden:
                    hidden = model.init_hidden(len(comments[1]))
                    predictions, attention, out_bool = model.forward(comments,hidden)
                    loss = loss_func(predictions.view(len(comments[1]), -1), targets.float())
                else:
                    predictions = model.forward(comments)
                    loss = loss_func(predictions, targets.float())
                running_loss += loss.item()
                all_preds.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

            metric_value = metric(np.vstack(all_targets), np.vstack(all_preds))
            return running_loss / len(data_generator), metric_value


import warnings
warnings.simplefilter("ignore", UserWarning)
def train(model, data_train, optimizer, loss_func, metric, data_val=None, n_epochs=10, evaluation=True):
    def progress_bar(rate, total=30):
        bar = '=' * int(total * rate)
        if int(total * rate) < total:
            bar += '>'
            bar += '-' * (total - int(total * rate) - 1)

        return '[' + bar + ']'
    
    print('Starting Training...', end='\r')
    train_size = len(data_train)
    train_losses = []    
    if evaluation:
        val_avg_losses = []
        val_scores = []
        train_scores = []
    
    start_time = time.time()
    best_score = 0
    for epoch in range(n_epochs):
        epoch_time = time.time()
        running_loss = 0

        if evaluation:
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

            if evaluation:
                all_preds.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        running_loss = running_loss / len(data_train)
        train_losses.append(running_loss)
        
        epoch_duration = time.time() - epoch_time
        avg_duration = (time.time() - start_time) / (epoch + 1)
        print(' ' * 100, end='\r')
        print('Epoch {} - avg_time/epoch: {:.2f}s - epoch_duration {:.2f}s'.format(epoch, avg_duration, epoch_duration))
        
        if evaluation:
            print('Evaluating the model on the validation set...', end='\r')
            t_score = metric(np.vstack(all_targets), np.vstack(all_preds))
            train_scores.append(t_score)
            if not data_val is None:
                avg_loss, v_score = evaluate(model, data_val, loss_func, metric)
                val_avg_losses.append(avg_loss)
                val_scores.append(v_score)
            
                print('    Train Loss: {:.3f}, Train Score {:.3f}, Val. Loss: {:.3f}, Val. Score: {:.3f}'.format(running_loss, 
                                                                                                             t_score, 
                                                                                                             avg_loss,
                                                                                                             v_score))
                if v_score > best_score:
                    best_score = v_score
                    utils.save_model(model, name=model.__class__.__name__)
                    print('    Saved Model as ' + '<<' + model.__class__.__name__ + '.txt>>.')
            
        else:
            print('    Train Loss: {:.2f}'.format(running_loss))
    
    if evaluation:
        return train_losses, train_scores, val_avg_losses, val_scores
    else:
        return train_losses

    
def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), 1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')
        
def train_attention (model, data_train, optimizer, loss_func, metric, I, penalization_coeff = 1, data_val=None, batch_size=32, n_epochs=10, evaluation=True):
    def progress_bar(rate, total=30):
        bar = '=' * int(total * rate)
        if int(total * rate) < total:
            bar += '>'
            bar += '-' * (total - int(total * rate) - 1)

        return '[' + bar + ']'
    
    print('Starting Training...', end='\r')
    train_size = len(data_train)
    train_losses = []    
    if evaluation:
        val_avg_losses = []
        val_scores = []
        train_scores = []
    
    start_time = time.time()
    
    for epoch in range(n_epochs): 
        model.train()
        total_loss = 0
        total_pure_loss = 0  # without the penalization term
        epoch_time = time.time()
        running_loss = 0
        
        if evaluation:
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
                
            
            hidden = model.init_hidden(len(inputs[1]))
            output, attention, out_bool = model.forward(inputs, hidden)
            # Remember that torch accumulates gradient
            optimizer.zero_grad()

            loss = loss_func(output.view(len(inputs[1]), -1),targets.float())
            total_pure_loss += loss.data.item()
            
            
            if out_bool:  # add penalization term
                attentionT = torch.transpose(attention, 1, 2).contiguous()
                extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
                loss += penalization_coeff * extra_loss
                
            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

            if evaluation:
                all_preds.append(output.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
        
        running_loss = running_loss / len(data_train)
        train_losses.append(running_loss)
        
        epoch_duration = time.time() - epoch_time
        avg_duration = (time.time() - start_time) / (epoch + 1)
        print('Epoch {} - avg_time/epoch: {:.2f}s - epoch_duration {:.2f}s'.format(epoch, avg_duration, epoch_duration))
        
        if evaluation:
            print('Evaluating the model on the validation set', end='\r')
            t_score = metric(np.vstack(all_targets), np.vstack(all_preds))
            train_scores.append(t_score)
            if not data_val is None:
                avg_loss, v_score = evaluate(model, data_val, loss_func, metric, True)
                val_avg_losses.append(avg_loss)
                val_scores.append(v_score)
            
                print('    Train Loss: {:.3f}, Train Score {:.3f}, Val. Loss: {:.3f}, Val. Score: {:.3f}'.format(running_loss, 
                                                                                                             t_score, 
                                                                                                             avg_loss,
                                                                                                             v_score))
        else:
            print('    Train Loss: {:.3f}'.format(running_loss))
    
    if evaluation:
        return train_losses, train_scores, val_avg_losses, val_scores
    else:
        return train_losses
    
def plot_result(results):
    plt.title('Loss Function')
    plt.plot(results[2], label='val')
    plt.plot(results[0], label='train')
    plt.legend()
    plt.show()
    
    plt.title('ROC_AUC Score')
    plt.plot(results[3], label='val')
    plt.plot(results[1], label='train')
    plt.legend()
    plt.show()


if __name__ == '__main__': 
    pass