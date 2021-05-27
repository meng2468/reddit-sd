import os
import time
import random
import numpy as np
import pandas as pd
import argparse

# torch
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

# huggingface
import transformers
from transformers import AutoTokenizer, AutoModel

# custom imports
from models.trainer import train, evaluate
from models.models import SimpleStDClassifier
from tools.processing import loadData, makeSplits

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


def main():
    parser = _makeParser()
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = True

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # data
    print("{:=^50}".format(" Loading data "))  
    all_data, fields = loadData(args.dataset, tokenizer, bs=args.bs, device=device)
    LABEL, TEXT, TARGET = fields

    distinct_targets = TARGET.vocab.itos[1:] # 0=<unk>
    macro = dict(loss=[], acc=[], fscore=[], precision=[], recall=[])

    # train one model per target
    for t, target in enumerate(distinct_targets, 1):
        print("{:=^50}".format(f" {target} ({t}/{len(distinct_targets)}) "))
        cache_dir = os.path.join(args.cache, target)
        print(f"Model will be saved in '{cache_dir}'")

        # make splits
        train_iter, test_iter = makeSplits(
            all_data, target, args.bs, device
            )        

        # pretrained model
        print(f"Loading {args.model}")
        base_model = AutoModel.from_pretrained(args.model)

        # init
        print("Initializing model")
        model = SimpleStDClassifier(base_model, len(LABEL.vocab))
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = nn.CrossEntropyLoss()
        model = model.to(device)
        criterion = criterion.to(device)

        # train
        print("Training")
        train(args.epochs, model, optimizer, criterion, train_iter, test_iter, 
            save_history=args.save, cache=cache_dir, verbose=args.verbose
            )

        # test
        print("Testing")
        metrics = evaluate(model, test_iter, criterion)
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
            print(f"{target}:", '   '.join(display))

    # macro evaluation
    print("{:=^50}".format(" Evaluation "))
    macro_display = [
        f"Loss {macro['loss'].mean():.2e}",
        f"Acc {macro['acc'].mean()*100:.2f}%",
        f"F1 {macro['fscore'].mean():.3f}",
        f"Precision {macro['precision'].mean():.3f}",
        f"Recall {macro['recall'].mean():.3f}",
    ] 
    print("MACRO: ", '   '.join(macro_display))

if __name__ == "__main__":
    main()