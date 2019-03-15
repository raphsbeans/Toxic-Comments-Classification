import torch
import time
import numpy as np
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from models import CoolNameNet
from data_loader import get_dataset, get_iterator, BatchGenerator


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
        else:
            print('    Train Loss: {:.2f}'.format(running_loss))
    
    if evaluation:
        return train_losses, train_scores, val_avg_losses, val_scores
    else:
        return train_losses


if __name__ == '__main__': 

    # First model -----------------------------------------------------------------------

    BATCH_SIZE = 32
    EMBEDDING_DIM = 100

    start_time = time.time()
    print('Loading dataset...', end='\r')
    print(' ' * 100, end='\r')
    train, val, test = get_dataset(vectors='glove.twitter.27B.100d')
    vocab = train.fields['comment_text'].vocab
    vectors = vocab.vectors.cuda()
    print('Dataset loaded - {:.2}s'.format(time.time() - start_time))    
    
    train_itr = BatchGenerator(get_iterator(train, BATCH_SIZE, train=True))
    val_itr = BatchGenerator(get_iterator(val, 256))
    test_itr = BatchGenerator(get_iterator(test, 256))

    model = CoolNameNet(len(vocab), EMBEDDING_DIM, 50, vectors=vectors, fine_tuning=False).cuda()
    out = model(torch.zeros((100, 32), dtype=torch.long).cuda())
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.BCELoss()

    results = model.train()



    # Second model -----------------------------------------------------------------------------