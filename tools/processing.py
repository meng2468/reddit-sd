import pandas as pd
import numpy as np
import spacy

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