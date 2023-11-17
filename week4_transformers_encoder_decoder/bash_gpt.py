import torch
from torch import nn

vocab_size = 16000
input_embedding_dimensions = 512
max_sequence_length = 1194
ff_dimension = 2048
num_heads=2
num_layers = 3
drop_rate = 0.1
device = "cuda:0" if torch.cuda.is_available() else 'cpu'


class positionalEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedding_dimensions = input_embedding_dimensions
        self.input_embeddings = nn.Embedding(vocab_size, input_embedding_dimensions)
        self.positional_encoding = nn.Embedding(max_sequence_length, input_embedding_dimensions)
    
    def forward(self, inputs):
        input_embeddings = self.input_embeddings(inputs)
        positional_embeddings = self.positional_encoding(torch.arange(inputs.size(1)).to(device))
        positional_embeddings = input_embeddings + positional_embeddings
        
        return positional_embeddings


class attention(torch.nn.Module):
    def __init__(self, is_causal = False):
        super().__init__()
        #MHA
        self.num_heads = num_heads
        self.head_dim = input_embedding_dimensions // self.num_heads
        self.is_causal = is_causal
        
        self.register_buffer('mask', torch.tril(torch.ones(max_sequence_length,max_sequence_length).view(1, 1,max_sequence_length,max_sequence_length)))
        self.mha_final_linear_layer = nn.Linear(input_embedding_dimensions,input_embedding_dimensions)
    
    def forward(self,positional_embeddings, q,k,v):

        heads_output = []
        heads_att_weights = []
        #input splitting (Q,K,V)
        batch_size = positional_embeddings.size(0)
        # q,k,v = self.mha_first_linear_layer(positional_embeddings).split(input_embedding_dimensions,dim=-1)
        # q = self.split_heads(q,batch_size)
        # k = self.split_heads(k,batch_size)
        # v = self.split_heads(v,batch_size)
        #MHA
        score_matrix = q @ k.permute(0,1,3,2)
        score_matrix = score_matrix/(self.head_dim ** 0.5)
        if self.is_causal:
            mask = self.mask == 0 #torch.tril(torch.ones(batch_size,1,score_matrix.size(2), score_matrix.size(3))) == 0
            mask = mask[:,:,:score_matrix.size(2),:score_matrix.size(2)]
            score_matrix = score_matrix.masked_fill(mask, float('-inf'))

        att_weights = torch.softmax(score_matrix,dim=-1)
        heads_att_weights.append(att_weights)

        output = att_weights @ v
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.head_dim * self.num_heads)

        heads_output.append(output)
        heads_output = torch.cat(heads_output, dim=-1)

        final_mha_output = self.mha_final_linear_layer(heads_output)
        return final_mha_output
    



class feedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_first_linear = nn.Linear(input_embedding_dimensions,ff_dimension)
        self.ff_relu = nn.ReLU()
        self.ff_second_linear = nn.Linear(ff_dimension,input_embedding_dimensions)
        self.drop   = torch.nn.Dropout(drop_rate)

    def forward(self, mha_norm_output):
        ff_first_linear_output = self.ff_first_linear(mha_norm_output)
        ff_relu_output = self.ff_relu(ff_first_linear_output)
        ff_output = self.ff_second_linear(ff_relu_output)
        ff_output = self.drop(ff_output)
        return ff_output


class transformer_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = attention()
        self.mha_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was mha_add_norm.size(-1) and i replaced with input_embedding_dimensions
        self.feedForward = feedForward()
        self.ff_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was ff_add_norm.size(-1) and i replaced with input_embedding_dimensions
    
    def forward(self, positional_embeddings, q,k,v):
        final_mha_output = self.attention(positional_embeddings, q,k,v)
        mha_add_norm = final_mha_output + positional_embeddings
        mha_norm_output = self.mha_norm_layer(mha_add_norm)
        ff_output = self.feedForward(mha_norm_output)

        ff_add_norm = ff_output + mha_norm_output
        ff_norm_output = self.ff_norm_layer(ff_add_norm)
        return ff_norm_output




class encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = attention()
        self.mha_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was mha_add_norm.size(-1) and i replaced with input_embedding_dimensions
        self.feedForward = feedForward()
        self.ff_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was ff_add_norm.size(-1) and i replaced with input_embedding_dimensions

    def forward(self, positional_embeddings, q,k,v):
        final_mha_output = self.attention(positional_embeddings, q,k,v)
        mha_add_norm = final_mha_output + positional_embeddings
        mha_norm_output = self.mha_norm_layer(mha_add_norm)
        ff_output = self.feedForward(mha_norm_output)

        ff_add_norm = ff_output + mha_norm_output
        ff_norm_output = self.ff_norm_layer(ff_add_norm)
        return ff_norm_output



class decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_attention = attention(is_causal=True)
        self.mha_first_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was mha_add_norm.size(-1) and i replaced with input_embedding_dimensions
        self.second_attention = attention(is_causal=False)
        self.mha_second_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was ff_add_norm.size(-1) and i replaced with input_embedding_dimensions
        self.feedForward = feedForward()
        self.ff_norm_layer = nn.LayerNorm(input_embedding_dimensions) # this was ff_add_norm.size(-1) and i replaced with input_embedding_dimensions
    
    def forward(self, positional_embeddings, q,k,v):
        first_attention_output = self.first_attention(positional_embeddings, q,k,v)
        first_mha_add_norm = first_attention_output + positional_embeddings
        first_mha_norm_output = self.mha_first_norm_layer(first_mha_add_norm)
        second_attention_output = self.second_attention(positional_embeddings, q,k,v)
        second_mha_add_norm = first_mha_norm_output + second_attention_output
        second_mha_norm_output = self.mha_second_norm_layer(second_mha_add_norm)
        ff_output = self.feedForward(second_mha_norm_output)

        ff_add_norm = ff_output + second_mha_norm_output
        ff_norm_output = self.ff_norm_layer(ff_add_norm)
        return ff_norm_output



class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_embedding_dimensions // self.num_heads
        self.enc_positional_embeddings = positionalEmbeddings()
        self.dec_positional_embeddings = positionalEmbeddings()
        self.mha_first_linear_layer = nn.Linear(input_embedding_dimensions,3*input_embedding_dimensions)
        self.drop    = torch.nn.Dropout(drop_rate)
        self.encoder = torch.nn.ModuleList([encoder() for _ in range(num_layers)])
        self.decoder = torch.nn.ModuleList([decoder() for _ in range(num_layers)])
        self.final_linear_layer = nn.Linear(input_embedding_dimensions,vocab_size)


    def forward(self, enc_inputs, dec_inputs):
        batch_size = enc_inputs.size(0)
        enc_positional_embeddings = self.enc_positional_embeddings(enc_inputs)
        enc_positional_embeddings = self.drop(enc_positional_embeddings)
        for encoder_layer in self.encoder:
            enc_q, enc_k, enc_v = self.qkv(batch_size,enc_positional_embeddings)
            enc_positional_embeddings  = encoder_layer(enc_positional_embeddings, enc_q, enc_k, enc_v)

        dec_positional_embeddings = self.dec_positional_embeddings(dec_inputs)
        dec_positional_embeddings = self.drop(dec_positional_embeddings)
        final_enc_q, final_enc_k, _ = self.qkv(batch_size,enc_positional_embeddings)
        for decoder_layer in self.decoder:
            _,_, dec_v = self.qkv(batch_size,dec_positional_embeddings)
            dec_positional_embeddings  = decoder_layer(dec_positional_embeddings, final_enc_q, final_enc_k, dec_v)    
        ff_norm_output = dec_positional_embeddings
        final_linear_layer_output = self.final_linear_layer(ff_norm_output)
        final_transformer_output = final_linear_layer_output #torch.softmax(final_linear_layer_output, dim=-1)
        return final_transformer_output
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def qkv(self,batch_size,positional_embeddings):
        q,k,v = self.mha_first_linear_layer(positional_embeddings).split(input_embedding_dimensions,dim=-1)
        q = self.split_heads(q,batch_size)
        k = self.split_heads(k,batch_size)
        v = self.split_heads(v,batch_size)
        return q,k,v


