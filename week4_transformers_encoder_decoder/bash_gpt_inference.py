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
    input_text = torch.tensor(tokenised_text)
    input_text = torch.LongTensor(input_text)
    input_text = input_text.view(1,-1)
    input_text = input_text.to(device)
    next_token_id = tk.PieceToId("[BOS]")
    iterable_text = torch.zeros(input_text.size(), dtype=torch.long)   
    iterable_text = torch.LongTensor(iterable_text)
    iterable_text[0,0] = next_token_id
    break_count = 0
    
    with torch.no_grad():
        while (next_token_id != tk.PieceToId("[EOS]")) & (break_count < input_text.size(1)-1):
            iterable_tensor = iterable_text
            iterable_tensor = iterable_tensor.view(1,-1)
            iterable_tensor = iterable_tensor.to(device)
            probability_matrix = inference_model(input_text,iterable_tensor)
            probability_matrix = torch.softmax(probability_matrix, dim=-1)
            probability_vector = torch.argmax(probability_matrix, dim=-1)
            print(probability_vector)
            # probability_vector = probability_matrix[0, -1, :]
            next_token_id = probability_vector[0,break_count]
            print(next_token_id)
            iterable_text[0,break_count+1] = next_token_id.item()
            break_count += 1
        
    final_text = tk.decode(iterable_text.tolist())
    return final_text

if __name__ == '__main__':
    if len(sys.argv) == 3:
        text = sys.argv[1]
        saved_model = sys.argv[2]
        
    print(inference(text, saved_model))

