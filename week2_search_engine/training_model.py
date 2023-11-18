import load_corpus
import generate_CBOW_ds
from datasets import load_dataset
import generate_two_tower_data
import pandas as pd
import sp_tokenizer
import torch
import two_tower_model
import wandb
import torch.optim as optim




def custom_collate_fn(batch):
    # Sort the batch by target sequence length (in descending order)
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    # Get the sequence lengths for both target and context
    target_seq_lengths = [len(item[0]) for item in batch]
    context_seq_lengths = [len(item[1]) for item in batch]

    # Pad target and context sequences separately
    padded_target_sequences = torch.nn.utils.rnn.pad_sequence([item[0] for item in batch], batch_first=True)
    padded_context_sequences = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True)

    # Create masks for the padded values for both target and context
    target_mask = (padded_target_sequences != 0).float()
    context_mask = (padded_context_sequences != 0).float()

    # Extract labels
    labels = [item[2] for item in batch]

    return torch.LongTensor(padded_target_sequences), torch.LongTensor(padded_context_sequences), torch.LongTensor(labels)

ds = torch.load('two_tower_training_data.pt', map_location='cpu')

batch_size = 512
dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

vocab_size = 16000  # Vocabulary size
embedding_dim = 100  # Dimension of word embeddings
hidden_dim = 128  # GRU hidden state size
output_dim = 64  # Desired output embedding dimension


model_GRU = two_tower_model.GRUEmbeddingModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model_LSTM = two_tower_model.LSTMEmbeddingModel(vocab_size, embedding_dim, hidden_dim, output_dim)
cosine_similarity_loss = two_tower_model.CosineSimilarityLoss(margin = 0.2)
optimizer = optim.SGD(list(model_GRU.parameters()) + list(model_LSTM.parameters()), lr=0.01)


wandb.login(key="9091f9245cf64052fa6d4eae03190a076fd87fe8")
wandb.init(project ='mlx_search_eninge', entity = 'basharkabalan',     config={
    "learning_rate": 0.01,
    "architecture": "two_tower",
    "dataset": "ms_marco_v1.1",
    "epochs": 3,
    })



num_epochs = 20
correct_predictions = 0
total_predictions = 0

for epoch in range(num_epochs):
    for query, doc, label in dl:
        optimizer.zero_grad()
        query = model_GRU(query)
        doc = model_LSTM(doc)
        # query = query.view(1,-1)
        # doc = doc.view(1,-1)
        loss,predictions = cosine_similarity_loss(query,doc,label)
        correct_predictions += torch.sum(predictions==label)
        total_predictions += batch_size
        acc = correct_predictions/total_predictions
        loss.backward()
        optimizer.step()


        wandb.log({"acc":acc,"loss": loss})
    print(f"Epoch {epoch+1}/{num_epochs}, Acc: {acc}, Loss: {loss.item()}")