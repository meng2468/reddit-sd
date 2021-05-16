import os
import torch
from torchtext.legacy import data
import pandas as pd
import numpy as np
import spacy



# ============= Reddit data processing =============
# Chooses relevant fields from scraped comments
def get_data(file_path):
    print('(1/4) Loading data from', file_path)
    df = pd.read_csv(file_path)
    df = df[['id','author','created_utc','score','subreddit','body']]
    return df

# Splits a single comment into multiple rows for each sentence
def convert_sentences(data, cores=4):
    print('(2/4) Splitting',len(data),'comments into sentences')
    texts = list(data['body'])
    nlp = spacy.load('en_core_web_sm', disable=['tagger','parser','lemmatizer', 'tok2vec', 'ner','attribute_ruler'])
    nlp.enable_pipe('senter')
    docs = nlp.pipe(texts, n_process = cores)
    sentences = []
    row = 0
    for doc in docs:
        sentences.append(list(doc.sents))
        row += 1
    data['sentences'] = sentences
    data = data.explode('sentences', ignore_index=True)
    data = data.rename(columns={'sentences':'text'})
    data['text'] = data['text'].apply(lambda x: ' '.join([str(word) for word in x]))
    return data.drop(columns='body')

# Removes non-keyword containing sentences
def filter_keywords(data, keywords):
    print('(3/4) Keeping sentences that contain', keywords)
    contain = data['text'].apply(lambda x: len(set(x.split()) & set(keywords)) > 0)
    return data[contain].reset_index(drop=True)

# Locate and keep complement and propositional clauses
def parse_clausal(data, cores=4):
    print('(4/4) Removing sentences without complement or propositional clauses')
    texts = list(data['text'])
    nlp = spacy.load('en_core_web_sm')
    docs = nlp.pipe(texts, n_process = cores)
    docs = list(docs)
    filter_l = []
    for doc in docs:
        keep = False
        for token in doc:
            if token.dep_ in ['ccomp', 'pcomp']:
                keep = True
        filter_l.append(keep)
    return data[filter_l].reset_index(drop=True)

# Run all pre-processing operations on file
def get_processed_data(file_path):
    print('*'*40)
    print('Starting data processing')
    df = get_data(file_path)
    sent_df = convert_sentences(df)
    key_df = filter_keywords(sent_df,['eth', 'Ethereum', 'ETH', 'Ether','ether'])
    par_df = parse_clausal(key_df)
    return par_df


# ============= Datasets processing  =============
def makeSplits(dataset, tokenizer, return_fields=False, **kwargs):
    """ Make splits from given dataset. Return train/val/test iterators. """
    print("*"*40)
    print(f"Making dataset splits ({dataset})")
    available_datasets = [ "SemEval2016Task6" ]
    assert dataset in available_datasets, "Invalid dataset, must be one of {}.".format(' | '.join(available_datasets))

    # data directory
    if os.path.isdir("./data/"):
        data_dir = "data"
    elif os.path.isdir("../data/"):
        data_dir = "../data"
    else:
        data_dir = None
    assert data_dir, "Please download dataset using 'datasets.sh' !"
    
    # informations
    dataset_info = {
        'SemEval2016Task6': dict(
            dir=f"{data_dir}/SemEval2016Task6/", 
            format="TSV",
            encoding="latin-1",
            train="trainingdata-all-annotations.txt", 
            val="trialdata-all-annotations.txt", 
            test="testdata-taskA-all-annotations.txt", 
            text=data.Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=128, pad_token=tokenizer.convert_tokens_to_ids(tokenizer.pad_token), 
                    unk_token=tokenizer.convert_tokens_to_ids(tokenizer.unk_token)), 
            label=data.Field(sequential=False, batch_first=True, dtype=torch.long, is_target=True),
            fields=("Stance", "Tweet")),

        'SemEval2019Task7': "TODO",
        }
    info = dataset_info[dataset]

    # encode files
    if "encoding" in info.keys():
        train = os.path.join(info['dir'], info['train'])
        val = os.path.join(info['dir'], info['val'])
        test = os.path.join(info['dir'], info['test'])
        new_train = os.path.join(info['dir'], "train.txt")
        new_val = os.path.join(info['dir'], "val.txt")
        new_test = os.path.join(info['dir'], "test.txt")

        encodeFile(train, new_train, info['encoding'], "utf-8")
        encodeFile(val, new_val, info['encoding'], "utf-8")
        encodeFile(test, new_test, info['encoding'], "utf-8")

        info['train'] = "train.txt"
        info['val'] = "val.txt"
        info['test'] = "test.txt"

    # read data & build vocab
    train_data, val_data, test_data = data.TabularDataset.splits(
        path=info['dir'], 
        train=info['train'], validation=info['val'], test=info['test'], 
        format=info['format'], 
        fields={info['fields'][0]: ('label', info['label']), info['fields'][1]: ('text', info['text'])}
        )
    info['label'].build_vocab(train_data)

    # make splits
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size=kwargs.get('bs', 32),
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        shuffle=True,
        device=kwargs.get('device', torch.device('cpu'))
    )
    if return_fields:
        return (train_iter, val_iter, test_iter), (info['label'], info['text'])
    return train_iter, val_iter, test_iter

# ============= Useful  =============
def encodeFile(input_file, output_file, input_enc="latin-1", output_enc="utf-8"):
    with open(output_file, 'w', encoding=output_enc) as new_f:
        with open(input_file, 'r', encoding=input_enc) as f:
            for line in f:
                new_f.write(line)
    print(f"Encoded {input_file} as {output_file} from {input_enc} into {output_enc}.")