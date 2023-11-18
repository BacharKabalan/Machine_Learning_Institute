
import torch
import pandas
import torch.optim as optim
# from datasets import load_dataset
import torch.nn.functional as F
import torch
import torch.nn as nn
import string
import numpy as np


class GRUEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        output, hidden = self.gru(embedded)
        embedded_sequence = self.fc(output)  # Using the last hidden state as the embedding
        embedded_sequence = torch.mean(embedded_sequence,dim=1)
        return embedded_sequence
    

class LSTMEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_sequence):
        embedded = self.embedding(input_sequence)
        output, (hidden, cell) = self.lstm(embedded)
        embedded_sequence = self.fc(output)  # Using the last hidden state as the embedding
        embedded_sequence = torch.mean(embedded_sequence,dim=1)
        return embedded_sequence
    

class CosineSimilarityLoss(nn.Module):
    def __init__(self, margin):
        super(CosineSimilarityLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, target, labels):
        cosine_similarity = F.cosine_similarity(predicted, target)
        predictions = torch.tensor(cosine_similarity>self.margin)
        loss = labels * (1-cosine_similarity) + (1-labels) * (1+cosine_similarity)
        return loss.mean(), predictions