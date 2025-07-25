import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
        ## scalling as per the paper

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, seq_len:int, dropout:float):
        super().__init__()
        #droput to prevent over fitting
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # matrix of shape seq_len X d_model
        pe = torch.zeros(seq_len,d_model) # one row for each position, column--posrtional embedding
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)  # seq_len,1 numerator  of the pe formula  squeezed to only get the sequence poition
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        # apply sine to even indices  0,2,4 ....
        pe[:, 0::2] = torch.sin(position * div_term)
        # cosine to odd  1,3,5 ....
        pe[:, 1::2] = torch.cos(position * div_term)
        # adding batch_dim
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)  # pe as a model state not as a trainal parameter
    def forward(self,x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    # features --> hidden_size or d_model to prevent internal covariete shift
    def __init__(self, features: int, eps: float=10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha-->learnable and multiplyied
        self.bias = nn.Parameter(torch.zeros(features))  # also learnable an offset
    def forward(self,x):
            # x-->(batch,seq_len,size)
            # dim--> for broadcasting
        mean = x.mean(dim = -1, keepdim = True) #(batch,seq_len,1)  512 dims --> single dim i.e mean of all 512 in sigle dim, by keeping keep _dims==true #For each token in each sentence, you collapse the 512 values into a single mean value.
        std = x.std(dim = -1, keepdim = True)#(batch,seq_len,1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff:int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model) # w2 and b2
    def forward(self,x):
        #flow of dimension-->> (batch,seq_len,d_model)--(batch,seq_len,d_ff)--(batch,seq_len,d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # Embedding vecctor sized
        self.h = h   # number of heads
        # check if d_model is divisble by h to get h number of q,k,v
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h # dimension of vector seen by each head d_model splits into h heads of dim d_k  SPLIT AXROSS DIM MEANS EAAVH HEAD WILL HABE AVESS TO EAVH SEQUENVE CUT A DIFF PART OF EMCEDDING
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)
        self.dropout = nn.Dropout(dropout)

    #defining the Attention Block
    @staticmethod
    def attention(query, key, value,mask,  dropout: nn.Dropout):
        d_k = query.shape[-1]   # fetching the d_k value
        #using the attention formula-->
        #(batch, h, seq_len, d_k)-->  (batch, h, seq_len, d_k)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        #attention_scores.shape = (batch, heads, seq_len_q, seq_len_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
            """For all positions where the mask is 0 (e.g., padding or future tokens), it sets the corresponding score to -1e9 → so softmax ≈ 0"""
        attention_scores = attention_scores.softmax(dim=-1) #(batch, h, seq_len_q, seq_Len_k)  # across seq_len_k
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    """(Lq, Lv/Lk) @ (Lv, d_k) → (Lq, d_k)--> final shape of (attention_scores @ value)is  output.shape = (B, h, Lq, d_k)
"""


    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(batch_Size, seq_len, d_model)-->(batch,seq_len,d_model) each token in q is multiplied to 512x512 dim matrix
        key = self.w_k(k)
        value = self.w_v(v)

        #(batch, seq_len, d_model) --> (batch, h, seq_len, d_k)   #molar heads
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        # calculating attention score
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        #(batch, h, seq_len, d_k)-transpose--> (batch, seq_len,h,d_k)--> batch,seq_len,d_model
        #.contiguous --> make sure that the tensors are contigous for .view
        #multiply with w_o
        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout) # applies dropout after the sublayers output implements layer norm
        self.norm = LayerNormalization(features)
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        # input x--> normalized --> passed to sublayer --> dropout applied --->this added back to orignal x

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float ):  # features--> d_model
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)]) # creatws 2 residual connection
    def forward(self, x, src_mask):
        #x-->input to the encoder block, src_mask for padding attention mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        #x--> input-->normalized-->multiheadattention_applied-->dropout_applied-->initial_x added
        x = self.residual_connections[1](x, self.feed_forward_block)
        #--> input x form multihead --> normalized --> feedforward_applied --> dropout-applied-->x added to residual
        return x

class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers # stacks encoder blocks
        #layers--> a list of EncoderBlock instances
        self.norm = LayerNormalization(features)
    def forward(self, x, mask):
        for layer in self.layers:
            # passing input through each encoder block sequentially
            x = layer(x, mask)
        return self.norm(x)  #normalize for stability


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.self_cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        #src_mask for input lan, tgt_mask for tarkgeted language mask
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.self_cross_attention_block(x,encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        #layer-->encoderblock
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    def forward(self, x):
        #conversion is like --> (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x) # output as raw logits no softmax  for CrossEntropyLoss
        # another approach--> returning torch.log_saoftmax(self.proj(x), dim = -1)
        # loss--> to be used NLLLoss as it requires logprobs ---------


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embd: InputEmbeddings,src_pos = PositionalEncoding, tgt_pos = PositionalEncoding, projection_layer = ProjectionLayer):
       super().__init__()
       self.encoder = encoder
       self.decoder = decoder
       self.src_embd = src_embed
       self.tgt_embd = tgt_embd
       self.src_pos = src_pos
       self.tgt_pos = tgt_pos
       self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        #(batch, seq_Len, d_model)
        src = self.src_embd(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        tgt = self.tgt_embd(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

#defining build_transformer funtion

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int = 6, h=8, dropout: float= 0.1, d_ff:int = 2048):
    #embedding layer
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    #positional encoding layer
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #defining the encoder block 6 in this case N=6
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    # create encoder, decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    # creating projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    # creating transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    # parameter initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer








