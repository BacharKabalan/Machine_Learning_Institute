# write a function          


def generate_corpus(df):
    corpus = []
    for index,record in df.iterrows():
        corpus.append(record['query'])
        corpus.extend(record['passages']['passage_text'])
    
    return corpus