#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[29]:


import sentencepiece as spm
import torch
import torch.nn as nn


# Importing the Transformer architecture

# In[30]:


from transformer import Transformer


# Write the inference function

# In[31]:


def inference(text, saved_model):
    vocab_size = 16000
    input_embedding_dimensions = 512
    max_sequence_length = 1194
    ff_dimension = 2048
    num_heads = 1
    
    # Instantiate the model
    inference_model = Transformer(vocab_size = vocab_size, input_embedding_dimensions = input_embedding_dimensions, max_sequence_length = max_sequence_length, ff_dimension=ff_dimension, num_heads = num_heads)
    # Step 3: Load the pre-trained weights
    inference_model.load_state_dict(torch.load(saved_model))
    # Set the model to evaluation mode (important for models with batch normalization and dropout)
    inference_model.eval()
    
    tk = spm.SentencePieceProcessor()
    tk.load('tinystories_tokeniser.model')
    
    tokenised_text = tk.EncodeAsIds(text)
    iterable_text = tokenised_text
    
    next_token_id = tk.PieceToId("<s>")
    break_count = 0
    with torch.no_grad():
        while (next_token_id != tk.PieceToId("</s>")) & (break_count <= max_sequence_length):
            iterable_tensor = torch.tensor(iterable_text)
            iterable_tensor = iterable_tensor.view(1,-1)
            
            probability_matrix = inference_model(iterable_tensor)
            #print(probability_matrix.size())
            probability_vector = probability_matrix[0, -1, :]
            #print(probability_vector.size())
            next_token_id = (torch.argmax(probability_vector))
            #print(next_token_id)
            iterable_text = iterable_text + [next_token_id.item()]
            #print(tk.decode(iterable_text))
            break_count += 1
        
    final_text = tk.decode(iterable_text)
    return final_text

