#!/usr/bin/env python
# coding: utf-8

# Import Libraries

# In[29]:


import sentencepiece as spm
import torch
import torch.nn as nn


# Importing the Transformer architecture

# In[30]:


import bash_gpt
import sys

# Write the inference function

# In[31]:


def inference(text, saved_model):
    max_sequence_length = 1194
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'
    # Instantiate the model
    inference_model = bash_gpt.GPT()
    saved_model = torch.load(saved_model)
    # Step 3: Load the pre-trained weights
    inference_model.load_state_dict(saved_model['model_state_dict'])
    inference_model = inference_model.to(device)
    # Set the model to evaluation mode (important for models with batch normalization and dropout)
    inference_model.eval()
    
    tk = spm.SentencePieceProcessor()
    tk.load('tiny_piece.model')
    
    tokenised_text = tk.EncodeAsIds(text)
    iterable_text = tokenised_text
    
    next_token_id = tk.PieceToId("[BOS]")
    break_count = len(iterable_text)
    
    while (next_token_id != tk.PieceToId("[EOS]")) & (break_count <= max_sequence_length):
        iterable_tensor = torch.tensor(iterable_text)
        iterable_tensor = iterable_tensor.view(1,-1)
        iterable_tensor = iterable_tensor.to(device)
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

if __name__ == '__main__':
    if len(sys.argv) == 3:
        text = sys.argv[1]
        saved_model = sys.argv[2]
        
    print(inference(text, saved_model))

