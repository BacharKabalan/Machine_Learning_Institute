#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import math
import torch.nn as nn


# In[5]:


class Transformer(torch.nn.Module):
  def __init__(self, vocab_size, input_embedding_dimensions, max_sequence_length, ff_dimension, num_heads):
    super(Transformer, self).__init__()
    #positional encoding
    self.input_embeddings = nn.Embedding(vocab_size, input_embedding_dimensions)
    self.positional_encoding = self.positional_encoding_function(max_sequence_length,input_embedding_dimensions)
    #MHA
    self.num_heads = num_heads
    self.head_dim = input_embedding_dimensions // self.num_heads
    self.mha_first_linear_layer = nn.Linear(input_embedding_dimensions,3*input_embedding_dimensions)
    self.mask = torch.tril(torch.ones(1,1,max_sequence_length, max_sequence_length))
    self.mha_final_linear_layer = nn.Linear(input_embedding_dimensions*self.num_heads,input_embedding_dimensions)
    #ADD&NORMALIZE
    
    self.mha_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was mha_add_norm.size(-1) and i replaced with input_embedding_dimensions
    #FEED FORWARD
    self.ff_first_linear = nn.Linear(input_embedding_dimensions,ff_dimension)
    self.ff_relu = nn.ReLU()
    self.ff_second_linear = nn.Linear(ff_dimension,input_embedding_dimensions)
    self.ff_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was ff_add_norm.size(-1) and i replaced with input_embedding_dimensions
    #final linear layer
    self.final_linear_layer = nn.Linear(input_embedding_dimensions,vocab_size)

  def positional_encoding_function(self, max_sequence_length, embedding_dimension):
    positional_embeddings = torch.zeros(max_sequence_length,embedding_dimension)
    position = torch.arange(0, max_sequence_length, dtype=torch.float)
    #transform it into a columned matrix with many rows and one column 
    position = position.unsqueeze(1)
    #then repeat it to have as many columns as embedding dimensions
    position = position.repeat(1,embedding_dimension)
    #do similar process for dimensions
    dimension = torch.arange(0,embedding_dimension, dtype=torch.float)
    dimension  = dimension.repeat(max_sequence_length,1)
    div_term = position/(10000)**(2*dimension/embedding_dimension)
    positional_embeddings[:,0::2] = torch.sin(div_term[:,0::2])
    positional_embeddings[:,1::2] = torch.cos(div_term[:,1::2])
    return positional_embeddings



  def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)



  def forward(self, inputs):
    #positional encoding
    input_embeddings = self.input_embeddings(inputs)
    positional_embeddings = input_embeddings + self.positional_encoding[:input_embeddings.size(1),:]
    heads_output = []
    batch_size = inputs.size(0)
    #input splitting (Q,K,V)
    for num_heads in range(self.num_heads):
      
      q,k,v = self.mha_first_linear_layer(positional_embeddings).split(input_embeddings.size(-1),dim=-1)
      q = self.split_heads(q,batch_size)
      k = self.split_heads(k,batch_size)
      v = self.split_heads(v,batch_size)
    #MHA
      score_matrix = q @ k.permute(0,1,3,2)
      score_matrix = score_matrix/(self.head_dim ** 0.5)
      mask = self.mask == 0#torch.tril(torch.ones(batch_size,1,score_matrix.size(2), score_matrix.size(3))) == 0
      mask = mask[:,:,:score_matrix.size(2),:score_matrix.size(2)]
      score_matrix = score_matrix.masked_fill(mask, float('-inf'))
      att_weights = torch.softmax(score_matrix,dim=-1)
      output = att_weights @ v
      output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)
      heads_output.append(output)
    heads_output = torch.cat(heads_output, dim=-1)
    final_mha_output = self.mha_final_linear_layer(heads_output)
    #ADD&NORMALIZE
    mha_add_norm = final_mha_output + positional_embeddings
    mha_norm_output = self.mha_norm_layer(mha_add_norm)
    #FEED FORWARD
    ff_first_linear_output = self.ff_first_linear(mha_norm_output)
    ff_relu_output = self.ff_relu(ff_first_linear_output)
    ff_output = self.ff_second_linear(ff_relu_output)
    ff_add_norm = ff_output + mha_norm_output
    ff_norm_output = self.ff_norm_layer(ff_add_norm)
    final_linear_layer_output = self.final_linear_layer(ff_norm_output)
    final_transformer_output = final_linear_layer_output #torch.softmax(final_linear_layer_output, dim=-1)

    return final_transformer_output

