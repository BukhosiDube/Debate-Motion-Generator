import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        
        assert d_model % num_heads == 0,
        
        #Initialize the dimensions
        self.d_model = d_model # Model's dimensions
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads
        
        #Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) #Query transformation
        self.W_k = nn.Linear(d_model, d_model) #Key transformation
        self.W_v = nn.Linear(d_model, d_model) #Value transformation
        self.W_o = nn.Linear(d_model, d_model) #output transformation
        
    def scaled_Dot_product_Attention(self, Q, K, V, mask=None):
        #Calculate the attention scores
        
        attention_Scores = torch.matmul(Q, K.transpose(-2, -1)) / mat.sqrt(self.d_k)
        
        #Apply mask if provided (prevents attention going to certain parts)
        if mask is not None:
            attention_Scores = attention_Scores.masked_fill(mask == 0, -1e9)
            
        #Softmax is applied to obtain attention probabilities
        attention_probs = torch.softmax(attention_Scores, dim=-1)
        
        output = torch.matmul(attention_probs, V)
        return output
    
    def split_heads(self, x):
        #Reshape the input to have num_heads for multi-head attention
        batch_size, _, seq_length, d_k = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        #Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
       #Applying the linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        #Performs scaled dot-product attention
        attention_output = self.scaled_Dot_product_Attention(Q, K, V, mask)
        
        #Combine heads and apply the output transformation
        output = self.W_o(self.combine_heads(attention_output))
        return output
    
class PositionWiseFeedForward(nn.Module): #Inherits the functionalities to work with neural network layers
    def __init__(self, d_model, d_ff):
        #d_model: dimensions of input and output
        #d_ff: Dimensions of inner layer in feed forward network
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_Seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_Seq_length, d_model)
        position = torch.arrange(0, max_Seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arrange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attention_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attention_dropout))
        attention_output = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_Seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_Embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_Embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_Seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout)for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout)for _ in range(num_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        Seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, Seq_length, Seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_Embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_Embedding(tgt)))
        
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
            
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        output = self.fc(dec_output)
        return output