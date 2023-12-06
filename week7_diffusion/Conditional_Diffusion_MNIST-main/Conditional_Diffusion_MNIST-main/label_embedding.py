import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA

def label_embedding(n_classes,dataset):

    
    # Step 1: Tokenization
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')
    
    # Step 2: Obtaining Embeddings
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        outputs = model(**tokenized_texts)
    
    # Extract the embeddings for each sentence
    embeddings = outputs.last_hidden_state[:, 0, :]  # Extract embeddings for [CLS] token
    
    # Step 3: Dimensionality Reduction with PCA
    n_components = min(n_classes, embeddings.shape[1])  # Set n_components to be less than or equal to the number of features
    pca = PCA(n_components=n_components)
    embeddings_reduced = pca.fit_transform(embeddings)
    
    return embeddings_reduced