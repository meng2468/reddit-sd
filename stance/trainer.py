import os
import time
import torch
import pandas as pd
import numpy as np

# Trainer functions
def train(max_epoch, model, optimizer, criterion, train_iterator, val_iterator, cache="../results", save_history=False):
    """ Trainer function for StDClassifier. """    
    # save history for later visualization
    history_df = pd.DataFrame({
        'type': [], # train or validation
        'epoch': [], # epoch no
        'loss': [],
        'acc': [],
        'fscore': [],
        'precision': [],
        'recall': [],
    })

    best_val_acc = -float('inf')
    for epoch in range(max_epoch):
        print(f"Epoch [{epoch+1:03}/{max_epoch}]")
        model.train() # make sure model is in train mode

        # timer
        start_time = time.time()

        # train one epoch
        train_metrics = _trainOneEpoch(model, train_iterator, optimizer, criterion)
        train_loss, train_acc, train_fscore, train_precision, train_recall = [v for v in train_metrics.values()]

        # evaluate
        val_metrics = evaluate(model, val_iterator, criterion)
        val_loss, val_acc, val_fscore, val_precision, val_recall = [v for v in val_metrics.values()]

        # stats
        duration = time.time() - start_time
        is_best = ((val_acc).mean() > best_val_acc)
        best_val_acc = max(best_val_acc, val_acc.mean()) # update best validation accuracy

        # history
        history_df = pd.concat([
            history_df, 
            pd.DataFrame({ **train_metrics, 'type': "train", 'epoch': epoch}), 
            pd.DataFrame({ **val_metrics, 'type': 'validation', 'epoch': epoch})
            ])
        
        # save checkpoint
        state = {
            'epoch': epoch + 1,
            'architecture': model.name,
            'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
            'best_acc': best_val_acc,
        }
        if save_history:
            state['history'] = history_df
        filename = os.path.normpath(f"{cache}/checkpoint.pth.tar")
        _saveCheckpoint(state, is_best, filename)

        # verbose
        display = [
            # f"Epoch [{epoch+1:03}/{max_epoch}]",
            f"Time {duration:.3f}s",
            f"Loss {train_loss.mean():.3e}",
            f"Acc {train_acc.mean()*100:.2f}%",
            f"F1 {train_fscore.mean():.3f}",
            f"Precision {train_precision.mean():.3f}",
            f"Recall {train_recall.mean():.3f}",
            
            f"Loss(val) {val_loss.mean():.2e}",
            f"Acc(val) {val_acc.mean()*100:.2f}%",
            f"F1(val) {val_fscore.mean():.3f}",
            f"Precision(val) {val_precision.mean():.3f}",
            f"Recall(val) {val_recall.mean():.3f}",
        ]
        print('   '.join(display))

def evaluate(model, iterator, criterion):
    epoch_metrics = {
        'loss': [], 
        'acc': [],
        'fscore': [],
        'precision': [],
        'recall': []
        }

    model.eval() # deactivate dropout

    with torch.no_grad():
        for batch in iterator:            
            logits = model.forward(batch.text)
            loss = criterion(logits, batch.label)
            acc = _accuracy(logits, batch.label)
            metrics = _metrics(logits, batch.label)
            fscore, precision, recall = [m for m in metrics.values()]

            epoch_metrics['loss'].append(loss.item())
            epoch_metrics['acc'].append(acc.item())
            epoch_metrics['fscore'].append(fscore.item())
            epoch_metrics['precision'].append(precision.item())
            epoch_metrics['recall'].append(recall.item())

    return { k:np.array(v) for k, v in epoch_metrics.items() } 

def _trainOneEpoch(model, iterator, optimizer, criterion):
    epoch_metrics = {
        'loss': [], 
        'acc': [],
        'fscore': [],
        'precision': [],
        'recall': []
        }
    
    start_time = time.time()
    total_op = len(iterator)
    for op, batch in enumerate(iterator):
        optimizer.zero_grad()
        
        logits = model.forward(batch.text) # text, text_lengths = batch.text
        loss = criterion(logits, batch.label)
        acc = _accuracy(logits, batch.label)
        metrics = _metrics(logits, batch.label)
        fscore, precision, recall = [m for m in metrics.values()]

        loss.backward()
        optimizer.step()

        epoch_metrics['loss'].append(loss.item())
        epoch_metrics['acc'].append(acc.item())
        epoch_metrics['fscore'].append(fscore.item())
        epoch_metrics['precision'].append(precision.item())
        epoch_metrics['recall'].append(recall.item())

        if (op-1) % 20  == 0: 
          duration = time.time() - start_time
          start_time = time.time()
          display = [
              "",
              f"[{op:03}/{total_op}]",
              f"{duration:.3f} s/op",
              f"Loss {np.mean(epoch_metrics['loss'][op:]):.3e}",
              f"F1 {np.mean(epoch_metrics['fscore'][op:]):.3f}",
          ]
          print('   '.join(display))        

    return { k:np.array(v) for k, v in epoch_metrics.items() } 


def _accuracy(y_pred, y_true):
    _, preds = torch.max(y_pred, dim=1)
    correct = torch.eq(preds, y_true)
    acc = correct.sum() / len(correct)
    return acc

def _metrics(y_pred, y_true, beta=1.0, tag=1.0):
    """ Computes metrics (fscore, precision & recall), can work with GPU. 
        Returns dict. 
    """
    _, preds = torch.max(y_pred, dim=1)

    favor = torch.eq(preds, tag) # if tag=0.0, it's actually against
    tfavor = torch.eq(y_true, tag)

    tp = (favor * tfavor).sum().to(torch.float32)
    tn = (~favor * ~tfavor).sum().to(torch.float32)
    fp = (favor * ~tfavor).sum().to(torch.float32)
    fn = (~favor * tfavor).sum().to(torch.float32)

    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    fscore = (beta**2+1) * (precision*recall) / ((beta**2)*precision + recall)

    return { 'fscore': fscore, 'precision': precision, 'recall': recall }

def _saveCheckpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(
            os.path.dirname(filename), 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)