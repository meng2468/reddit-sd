import os
import json
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
    os.makedirs(cache_dir, exist_ok=True)

    if data_args.dataset_name == "SemEval2016Task6":
        # filenames
        trainfile = os.path.join(data_dir, "trainingdata-all-annotations.txt")
        testfile = os.path.join(data_dir, "testdata-taskA-all-annotations.txt")
        
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
        new_trainfile = os.path.join(cache_dir, "train.csv")
        new_testfile = os.path.join(cache_dir, "test.csv")
        train_data.to_csv(new_trainfile, sep=',', encoding='utf-8')
        test_data.to_csv(new_testfile, sep=',', encoding='utf-8')

        # create dataset
        data_files = {'train': new_trainfile, 'validation': new_testfile, 'test': new_testfile}
        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        return datasets
    
    elif data_args.dataset_name == "ARC":
        bodyfile = os.path.join(data_dir, "arc_bodies.csv")
        trainfile = os.path.join(data_dir, "arc_stances_train.csv")
        testfile = os.path.join(data_dir, "arc_stances_test.csv")

        # format data using pandas
        bodies = pd.read_csv(bodyfile)
        train_data = pd.read_csv(trainfile).merge(bodies, how='left', on='Body ID')
        test_data = pd.read_csv(testfile).merge(bodies, how='left', on='Body ID')
        useful_columns = ["Headline", "Stance", "articleBody"]
        renamed_columns = {'articleBody': "text", 'Stance': "label", 'Headline': "target"}
        train_data = train_data.loc[:, useful_columns].rename(columns=renamed_columns)
        test_data = test_data.loc[:, useful_columns].rename(columns=renamed_columns)

        # save newly created data
        new_trainfile = os.path.join(cache_dir, "train.csv")
        new_testfile = os.path.join(cache_dir, "test.csv")
        train_data.to_csv(new_trainfile, sep=',', encoding='utf-8')
        test_data.to_csv(new_testfile, sep=',', encoding='utf-8')

        # create dataset
        data_files = {'train': new_trainfile, 'validation': new_testfile, 'test': new_testfile}
        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        return datasets
    
    elif data_args.dataset_name == "FNC-1":
        trainbodyfile = os.path.join(data_dir, "train_bodies.csv")
        testbodyfile = os.path.join(data_dir, "competition_test_bodies.csv")
        trainfile = os.path.join(data_dir, "train_stances.csv")
        testfile = os.path.join(data_dir, "competition_test_stances.csv")

        # format data using pandas
        trainbodies = pd.read_csv(trainbodyfile)
        testbodies = pd.read_csv(testbodyfile)
        train_data = pd.read_csv(trainfile).merge(trainbodies, how='left', on='Body ID')
        test_data = pd.read_csv(testfile).merge(testbodies, how='left', on='Body ID')
        useful_columns = ["Headline", "Stance", "articleBody"]
        renamed_columns = {'articleBody': "text", 'Stance': "label", 'Headline': "target"}
        train_data = train_data.loc[:, useful_columns].rename(columns=renamed_columns)
        test_data = test_data.loc[:, useful_columns].rename(columns=renamed_columns)

        # save newly created data
        new_trainfile = os.path.join(cache_dir, "train.csv")
        new_testfile = os.path.join(cache_dir, "test.csv")
        train_data.to_csv(new_trainfile, sep=',', encoding='utf-8')
        test_data.to_csv(new_testfile, sep=',', encoding='utf-8')

        # create dataset
        data_files = {'train': new_trainfile, 'validation': new_testfile, 'test': new_testfile}
        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        return datasets

    elif data_args.dataset_name == "PERSPECTRUM":
        # load data
        tgt_file = os.path.join(data_dir, "perspective_pool_v1.0.json")
        targets = pd.read_json(tgt_file)
        
        split_file = os.path.join(data_dir, "dataset_split_v1.0.json")
        splits = json.load(open(split_file, 'r'))
        
        stances_file = os.path.join(data_dir, "perspectrum_with_answers_v1.0.json")
        stances = pd.json_normalize(
            data=json.load(open(stances_file, 'r')), 
            record_path=["perspectives", "pids"], 
            meta=["text", "source", "cId", ["perspectives", "stance_label_3"]], 
            record_prefix="pId"
        )

        # format data using pandas
        all_data = stances.merge(targets, how='left', left_on='pId0', right_on='pId', suffixes=('_stance', '_target'))
        all_data['split'] = all_data['cId'].astype(str).map(splits)

        useful_columns = ["text_stance", "text_target", "perspectives.stance_label_3", "split"]
        renamed_columns = {'text_stance': "text", 'text_target': "target", 'perspectives.stance_label_3': "label"}
        all_data = all_data.loc[:, useful_columns].rename(columns=renamed_columns)

        # save newly created data
        new_trainfile = os.path.join(cache_dir, "train.csv")
        new_valfile = os.path.join(cache_dir, "validation.csv")
        new_testfile = os.path.join(cache_dir, "test.csv")
        train_data = all_data.loc[all_data.split == 'train', ["text", "target", "label"]]
        val_data = all_data.loc[all_data.split == 'dev', ["text", "target", "label"]]
        test_data = all_data.loc[all_data.split == 'test', ["text", "target", "label"]]
        train_data.to_csv(new_trainfile, sep=',', encoding='utf-8')
        val_data.to_csv(new_valfile, sep=',', encoding='utf-8')
        test_data.to_csv(new_testfile, sep=',', encoding='utf-8')

        # create dataset
        data_files = {'train': new_trainfile, 'validation': new_valfile, 'test': new_testfile}
        datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir)
        return datasets

    else:
        raise NotImplementedError("Invalid dataset name, work in progress...")
