import sentencepiece as spm
import torch
import random

def generate_two_tower_data(train_df,sp_model):
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model)
    passage_dict = {}
    store = []
    for index,record in train_df.iterrows():
        query_text = record['query']
        query_tokens = sp.encode_as_pieces(query_text)
        # Convert tokens to token IDs
        query_token_ids = [sp.piece_to_id(token) for token in query_tokens]

        for doc_text in record['passages']['passage_text']:
            doc_tokens = sp.encode_as_pieces(doc_text)
            # Convert tokens to token IDs
            doc_token_ids = [sp.piece_to_id(token) for token in doc_tokens]
            store.append((torch.LongTensor(query_token_ids),torch.LongTensor(doc_token_ids),torch.tensor(1)))
            passage_dict[doc_text]  = torch.LongTensor(doc_token_ids)

        negative_index = random.choice(train_df.index[train_df.index != index])
        for doc_text in train_df.loc[negative_index,'passages']['passage_text']:
            doc_tokens = sp.encode_as_pieces(doc_text)
            # Convert tokens to token IDs
            doc_token_ids = [sp.piece_to_id(token) for token in doc_tokens]
            store.append((torch.LongTensor(query_token_ids),torch.LongTensor(doc_token_ids),torch.tensor(0)))


    return store, passage_dict