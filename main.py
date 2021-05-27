import os
import time
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import argparse

# custom imports
from stance.trainer import train, evaluate
from stance.models import SimpleStDClassifier
from tools.processing import makeSplits, targetIterator, getDistinctTargets

def _makeParser():
    available_models = [ "bert-base-uncased", "albert-base-v2", "distilbert-base-uncased", "roberta-base" ]
    available_datasets = [ "SemEval2016Task6" ]

    parser = argparse.ArgumentParser(description="Stance Detection benchmark")
    # required arguments
    parser.add_argument('model', type=str, metavar='BASE_MODEL', choices=available_models, 
        help="pretrained base model, must be one of {}".format(' | '.join(available_models)))
    parser.add_argument('dataset', type=str, metavar='DATASET', choices=available_datasets, 
        help="dataset name, must be one of {}".format(' | '.join(available_datasets)))
    
    # optional arguments
    parser.add_argument('--drop', type=float, default=0, help="dropout rate")
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='max epochs')

    parser.add_argument('--seed', type=int, default=None, help='optional seed')
    parser.add_argument('--cache', type=str, default="./results", help='cache directory to save models')
    parser.add_argument('--save', action='store_true', help="save training history")
    parser.add_argument('-v', '--verbose', action='store_true', help="print log messages")
    return parser


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained models
    print("{:=^50}".format(" Loading PLMs "))  
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    base_model = AutoModel.from_pretrained(config['model_name'])

    # data
    print("{:=^50}".format(" Loading data "))  
    iterators, fields = makeSplits(config['dataset'], tokenizer, return_fields=True, bs=config['bs'], device=device)
    train_iter, val_iter, test_iter = iterators
    LABEL, TEXT, TARGET = fields 
    # train_iter, val_iter, test_iter = makeSplits(config['dataset'], tokenizer, **config)

    # init
    print("{:=^50}".format(" Initialization "))  
    # model = StDClassifierWithTargetSpecificHeads(base_model, len(LABEL.vocab), heads=len(TARGET.vocab))
    model = SimpleStDClassifier(base_model, len(LABEL.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # train
    print("{:=^50}".format(" Training "))  
    train(config['max_epoch'], model, optimizer, criterion, train_iter, val_iter, save_history=True)

    # test
    print("{:=^50}".format(" Evaluation "))  
    macro = dict(loss=[], acc=[], fscore=[], precision=[], recall=[])

    # for target in range(len(TARGET.vocab)):
    #     target_name = TARGET.vocab.itos[target]
    distinct_targets = getDistinctTargets(train_iter, tokenizer)
    for target_name, target in distinct_targets.items():
        target_iter = targetIterator(train_iter, target)
        metrics = evaluate(model, target_iter, criterion)
        loss, acc, fscore, precision, recall = [v[~np.isnan(v)] for v in metrics.values()]

        if loss.shape[0] > 0:
            # append to macro scores
            for k, v in metrics.items(): 
                macro[k] = np.concatenate([macro[k], v[~np.isnan(v)]])

            # verbose
            display = [
                f"Loss {loss.mean():.2e}",
                f"Acc {acc.mean()*100:.2f}%",
                f"F1 {fscore.mean():.3f}",
                f"Precision {precision.mean():.3f}",
                f"Recall {recall.mean():.3f}",
            ]
            print(f"{target_name.title()}:", '   '.join(display))

    macro_display = [
        f"Loss {macro['loss'].mean():.2e}",
        f"Acc {macro['acc'].mean()*100:.2f}%",
        f"F1 {macro['fscore'].mean():.3f}",
        f"Precision {macro['precision'].mean():.3f}",
        f"Recall {macro['recall'].mean():.3f}",
    ] 
    print("MACRO: ", '   '.join(macro_display))

if __name__ == "__main__":
    bert()