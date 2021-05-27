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
def loadData(dataset, tokenizer, **kwargs):
    """ Make splits from given dataset. Return train/test iterators. """
    print(f"Making dataset splits ({dataset})")
    available_datasets = [ "SemEval2016Task6", "ARC" ]
    assert dataset in available_datasets, "Invalid dataset, must be one of {}.".format(' | '.join(available_datasets))

    # data directory
    basedir = os.path.dirname(os.path.realpath(__file__))
    parentdir = os.path.dirname(basedir)
    data_dir = os.path.join(parentdir, "data")
    assert len(os.listdir(data_dir)) > 0, "Please download dataset using 'datasets.sh' !"

    # fields
    TEXT = data.Field(
        use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
        fix_length=128, pad_token=tokenizer.convert_tokens_to_ids(tokenizer.pad_token), 
        unk_token=tokenizer.convert_tokens_to_ids(tokenizer.unk_token))
    LABEL = data.Field(sequential=False, batch_first=True, dtype=torch.long, is_target=True)
    TARGET = data.Field(sequential=False, batch_first=True, dtype=torch.long)
    
    # TARGET = data.Field(
    #     use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
    #     fix_length=32, pad_token=tokenizer.convert_tokens_to_ids(tokenizer.pad_token), 
    #     unk_token=tokenizer.convert_tokens_to_ids(tokenizer.unk_token))
    
    # preprocess
    if dataset == "SemEval2016Task6":
        path = f"{data_dir}/SemEval2016Task6/"

        # encode files
        encoding = "latin-1"
        train = os.path.join(path, "trainingdata-all-annotations.txt")
        test = os.path.join(path, "testdata-taskA-all-annotations.txt")
        new_train = os.path.join(path, "train.txt")
        new_test = os.path.join(path, "test.txt")

        encodeFile(train, new_train, encoding, "utf-8")
        encodeFile(test, new_test, encoding, "utf-8")

        fields = {
            "Stance": ('label', LABEL), 
            "Tweet": ('text', TEXT),
            "Target": ('target', TARGET)
            }

    elif dataset == "ARC":
        path = f"{data_dir}/ARC/"
        bodyfile = os.path.join(path, "arc_bodies.csv")
        trainfile = os.path.join(path, "arc_stances_train.csv")
        testfile = os.path.join(path, "arc_stances_test.csv")

        # load data
        bodies = pd.read_csv(bodyfile)
        train_data = pd.read_csv(trainfile).merge(bodies, how='left', on='Body ID')
        test_data = pd.read_csv(testfile).merge(bodies, how='left', on='Body ID')   

        # save newly created data
        train_data.to_csv(os.path.join(path, "train.txt"), sep='\t')
        test_data.to_csv(os.path.join(path, "test.txt"), sep='\t')

        # fields
        fields = {
            "Stance": ('label', LABEL), 
            "articleBody": ('text', TEXT),
            "Headline": ('target', TARGET)
            }

    else:
        raise NotImplementedError("Invalid dataset name")

    # load data into torch Dataset (tabular dataset)
    train_data, test_data = data.TabularDataset.splits(
        path=path, 
        train="train.txt", test="test.txt", 
        format="tsv", 
        fields=fields
    )

    # build vocab
    LABEL.build_vocab(train_data)
    TARGET.build_vocab(train_data)

    # return data and fields
    return (train_data, test_data), (LABEL, TEXT, TARGET)

def makeSplits(all_data, target, bs, device):
    all_train_data, all_test_data = all_data

    # new iterators
    train_data = [s for s in all_train_data if s.target == target]
    test_data = [s for s in all_test_data if s.target == target]

    # convert to torchtext Dataset
    train_data = data.Dataset(train_data, all_train_data.fields)
    test_data = data.Dataset(test_data, all_test_data.fields)

    # make splits
    train_iter, test_iter = data.BucketIterator.splits(
        (train_data, test_data), 
        sort=False,
        batch_size=bs,
        shuffle=True,
        device=device
    )
    return train_iter, test_iter

# ============= Useful  =============
def encodeFile(input_file, output_file, input_enc="latin-1", output_enc="utf-8"):
    with open(output_file, 'w', encoding=output_enc) as new_f:
        with open(input_file, 'r', encoding=input_enc) as f:
            for line in f:
                new_f.write(line)
    print(f"Encoded {input_file} as {output_file} from {input_enc} into {output_enc}.")

def targetIterator(iterator, target):
    """ Transforms the iterator to match a specific target """
    for batch in iterator:
        mask = (batch.target==target).prod(dim=1)
        batch.text = batch.text[mask]
        batch.target = batch.target[mask]
        batch.label = batch.label[mask]
        if batch.text.shape[0] > 0: # skip empty batches
            yield batch
        else:
            continue
  
def getDistinctTargets(iterator, tokenizer):
    distinct_targets = {}
    for batch in iterator:
        tgt = batch.target[0]
        ids = tgt[tgt != 0] # remove [PAD]
        ids = ids.tolist()[1:-1] # remove [CLS] and [SEP]

        tokens = tokenizer.convert_ids_to_tokens(ids) 
        target = tokenizer.convert_tokens_to_string(tokens)

        if target not in distinct_targets:
            distinct_targets[target] = tgt
    
    return distinct_targets