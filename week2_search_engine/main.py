
import generate_corpus
from datasets import load_dataset
import pandas as pd
import sentencepiece as spm
import torch
import random


dataset = load_dataset("ms_marco", "v1.1")
train_df = dataset['train']
val_df = dataset['validation']
test_df = dataset['test']
dataset_df = pd.concat([train_df,val_df,test_df],axis = 0)
corpus = generate_corpus.generate_corpus(dataset_df)
print(corpus[0])

