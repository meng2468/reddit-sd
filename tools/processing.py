import pandas as pd
import numpy as np
import spacy

# Chooses relevant fields from scraped comments
def get_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['author','created_utc','score','subreddit','body']]
    return df

# Splits a single comment into multiple rows for each sentence
def convert_sentences(data, cores=4):
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
    contain = data['text'].apply(lambda x: len(set(x.split()) & set(keywords)) > 0)
    return data[contain].reset_index()

# Locate complement clauses
def parse_extract(data, cores=4):
    texts = list(data['text'])
    nlp = spacy.load('en_core_web_sm', disable=['tagger','lemmatizer', 'tok2vec', 'ner','attribute_ruler'])
    docs = nlp.pipe(texts, n_process = cores)
    