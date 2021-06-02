import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset

def load_custom_dataset(data_args, cache_dir=None):
    cache_dir = "~/datasets" if cache_dir is None else cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    # data directory
    base_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    parent_dir = Path(os.path.dirname(base_dir))
    data_dir = parent_dir/"data"
    assert len(os.listdir(data_dir)) > 0, "Please download dataset using 'datasets.sh' !"
    data_dir = data_dir/data_args.dataset_name

    if data_args.dataset_name == "SemEval2016Task6":
        # filenames
        trainfile = os.path.join(data_dir, "trainingdata-all-annotations.txt")
        testfile = os.path.join(data_dir, "testdata-taskA-all-annotations.txt")
        new_trainfile = os.path.join(cache_dir, "train.csv")
        new_testfile = os.path.join(cache_dir, "test.csv")
        
        # format data using pandas
        train_data = pd.read_csv(trainfile, sep='\t', encoding='latin-1') 
        test_data = pd.read_csv(testfile, sep='\t', encoding='latin-1') 

        useful_columns = ["Tweet", "Stance", "Target"]
        train_data = train_data.loc[:, useful_columns]
        test_data = test_data.loc[:, useful_columns]

        renamed_columns = {'Tweet': "text", 'Stance': "label", 'Target': "target"}
        train_data.rename(columns=renamed_columns, inplace=True)
        test_data.rename(columns=renamed_columns, inplace=True)

        # save formatted files
        os.makedirs(cache_dir, exist_ok=True)
        train_data.to_csv(new_trainfile, sep=',', encoding='utf-8')
        test_data.to_csv(new_testfile, sep=',', encoding='utf-8')

        # create dataset
        data_files = {'train': new_trainfile, 'validation': new_testfile, 'test': new_testfile}
        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        return datasets

    else:
        raise NotImplementedError("Invalid dataset name, work in progress...")
