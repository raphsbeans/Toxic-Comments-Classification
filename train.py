import torch
import time
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def evaluate(model, data_generator, loss_func, metric):
    with torch.no_grad():
        running_loss = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for comments, targets in data_generator:
                predictions = model.forward(comments)
                loss = loss_func(predictions, targets.float())
                running_loss += loss.item()
                all_preds.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())

            metric_value = metric(np.vstack(all_targets), np.vstack(all_preds))
            return running_loss / len(data_generator), metric_value


def train(model, data_train, optimizer, loss_func, metric, data_val=None, n_epochs=10, evaluation=True, loss_period=100):
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
    running_loss = 0 # total loss since last append
    total_batchs = 0 # total number of batch 
    for epoch in range(n_epochs):        
        epoch_time = time.time()        

        if evaluation:
            all_preds = []
            all_targets = []

        counter = 0 #number of batchs in this epoch
        for inputs, targets in data_train:
            total_batchs += 1
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

            if total_batchs % loss_period:
                running_loss = running_loss / loss_period
                train_losses.append(running_loss)
                running_loss = 0
                
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
        else:
            print('    Train Loss: {:.2f}'.format(running_loss))
    
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