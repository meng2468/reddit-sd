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
from stance.models import StDClassifier
from tools.processing import makeSplits

def _makeParser():
    available_models = [ "bert-base-uncased" ]
    available_datasets = [ "SemEval2016Task6" ]

    parser = argparse.ArgumentParser(description="Stance Detection benchmark")
    parser.add_argument('model', type=str, metavar='BASE_MODEL', choices=available_models, 
        help="pretrained base model, must be one of {}".format(' | '.join(available_models)))
    parser.add_argument('dataset', type=str, metavar='DATASET', choices=available_datasets, 
        help="dataset name, must be one of {}".format(' | '.join(available_datasets)))

def bert():
    # config
    config = {
        'model_name': "bert-base-uncased",
        'dataset': "SemEval2016Task6",
        'max_epoch': 5,
        'lr': 0.0001, # learning rate
        'bs': 16, # batch size
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # pretrained models
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    base_model = AutoModel.from_pretrained(config['model_name'])

    # data
    iterators, fields = makeSplits(config['dataset'], tokenizer, return_fields=True, bs=config['bs'], device=device)
    train_iter, val_iter, test_iter = iterators
    LABEL, TEXT = fields 
    # train_iter, val_iter, test_iter = makeSplits(config['dataset'], tokenizer, **config)

    # init
    model = StDClassifier(base_model, len(LABEL.vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    criterion = criterion.to(device)

    # train
    train(config['max_epoch'], model, optimizer, criterion, train_iter, val_iter, save_history=True)

    # test
    metrics = evaluate(model, optimizer, criterion, test_iter)
    loss, acc, fscore, precision, recall = [v for v in metrics.values()]


if __name__ == "__main__":
    bert()