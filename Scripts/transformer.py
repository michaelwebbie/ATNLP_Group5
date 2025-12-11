
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()

        #input is batch_size, seq_len and emb_dim (e.g., 32 sentences, 10 tokens each, each token a 512-dim vector)

        assert emb_dim % num_heads == 0 #must be divisible
        self.emb_dim = emb_dim #length of vector for one token (e.g., 512)
        self.num_heads = num_heads #number of attention heads (e.g., 8)
        self.head_dim = emb_dim // num_heads #embedding information is split across heads (e.g., 512-dim vector/8 heads = 64 dims per head) - encourages specialisation and helps generalisation

        #Create weight matrices for query, key, value
        # each takes in an embedding vector (e.g. 512) and projects it into a new vector of size num_heads * head_dim (equal to emb_dim), using learnable weights
        #Each are different representations that can be learned through the model
        self.query_linear = nn.Linear(emb_dim, num_heads * self.head_dim) #Weights learn 'what am I interested in?'
        self.key_linear   = nn.Linear(emb_dim, num_heads * self.head_dim) #'What information do I contain that others might want?'
        self.value_linear = nn.Linear(emb_dim, num_heads * self.head_dim) #'If someone chooses me, what info do I actually give them?'

        self.out_linear = nn.Linear(num_heads * self.head_dim, emb_dim) # Final projection to recombine information from all heads at the end

    def forward(self, query, key, value, mask=None):
        batch_size, query_len, _ = query.shape
        _, key_len, _ = key.shape

        #Apply linear layers from above
        #For each token, multiply by the weight matrix and produce same size output as the embedding
        Q = self.query_linear(query)
        K = self.key_linear(key)
        V = self.value_linear(value)

        #Splits emb_dim into num_heads * head_dim to prepare for input into each head
        #transpose changes order to (batch_size, num_heads, seq_len, head_dim) - attention is computed per head
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key_len,   self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, key_len,   self.num_heads, self.head_dim).transpose(1, 2)

        # Transpose K so that matrix multiplication is possible (query_len × head_dim) × (head_dim × key_len)
        # Matrix multiplication computes similarity scores between each query token and each key token
        #Key_out: how much should token care about every other token in the sequence?
        key_out = torch.matmul(Q, K.transpose(-2, -1))

        #Masks padding tokens and future tokens (so model does not cheat)
        #Sets key_out value for masks to -1e20 (huge negative number)
        if mask is not None:
            key_out = key_out.masked_fill(mask == 0, -1e20)

        #Scale by sqrt(head_dim) to stabilise gradients
        #Apply softmax to get probabilities
        #Huge negatives (masks) get 0 probability
        attention = torch.softmax(
            key_out / math.sqrt(self.head_dim),
            dim=-1)

        # matrix multiplication of value
        #Takes attention weights (importance of tokens to each other) and take a weighted average of value vectors,
        #Producing a context-aware representation for each token
        out = torch.matmul(attention, V)

        # Reverses transposition from earlier (swaps num_heads and query_len back)
        out = out.transpose(1, 2).contiguous()

        #Concatenate the heads
        #Shape back to (batch_size, seq_len, emb_dim)
        out = out.view(batch_size, query_len, self.num_heads * self.head_dim)

        #Produces final representation from concatenated heads
        out = self.out_linear(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, forward_dim):
        super().__init__()

        #Uses attention heads as defined above
        self.attention = MultiHeadAttention(emb_dim, num_heads)

        #After applying attnetion, token passed through NN to compute new non-linear features from the embedding
        #Forward_dim > emb_dim to learn more complex features
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, forward_dim),
            nn.ReLU(),
            nn.Linear(forward_dim, emb_dim)
        )

        # Normalisation layers - rescales embeddings
        self.norm1 = nn.LayerNorm(emb_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(emb_dim, eps=1e-6)

        # Dropout for generalisation
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        #Multi-head attention
        attention_out = self.attention(query, key, value, mask)

        # Normalization + dropout
        x = self.norm1(self.dropout(attention_out + query))

        # Neural Network
        ffn_out = self.ffn(x)

        # Normalization + dropout
        out = self.norm2(self.dropout(ffn_out + x))

        return out

def get_sinusoid_table(max_len, emb_dim):
    def get_angle(pos, i, emb_dim):
        return pos / 10000 ** ((2 * (i // 2)) / emb_dim)

    sinusoid_table = torch.zeros(max_len, emb_dim)
    for pos in range(max_len):
        for i in range(emb_dim):
            if i % 2 == 0:
                sinusoid_table[pos, i] = math.sin(get_angle(pos, i, emb_dim))
            else:
                sinusoid_table[pos, i] = math.cos(get_angle(pos, i, emb_dim))
    return sinusoid_table

class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()

        # Create embedding lookup table
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # Positional encodings (sinusoidal)
        # Ensures the position of each token in a sentence can stay intact
        sinusoid_table = get_sinusoid_table(max_len + 1, emb_dim)
        self.position_embedding = nn.Embedding.from_pretrained(
            sinusoid_table, freeze=True)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer block from above
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    forward_dim=forward_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        batch_size, seq_len = x.shape

        # Create positions for each token in a sentence
        #+1 so positioning starts at 1 rather than 0
        positions = torch.arange(1, seq_len + 1).unsqueeze(0).expand(batch_size, seq_len)
        positions = positions.to(x.device)

        # Token + positional embeddings
        out = self.token_embedding(x) + self.position_embedding(positions)

        # Apply dropout
        out = self.dropout(out)

        # Pass through Transformer blocks
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, forward_dim, dropout):
        super().__init__()

        # Attention layer from above
        self.self_attention = MultiHeadAttention(emb_dim, num_heads)

        #Normalization layer
        self.norm = nn.LayerNorm(emb_dim, eps=1e-6)

        #Transformer block
        self.transformer_block = TransformerBlock(
            emb_dim=emb_dim,
            num_heads=num_heads,
            dropout=dropout,
            forward_dim=forward_dim
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tgt_mask):
        #src_mask - masks padding
        #tgt_mask - masks future decoder tokens

        #Attention including tgt_mask so that future tokens are masked
        self_attn = self.self_attention(x, x, x, tgt_mask)

        #Normalisation and dropout
        query = self.norm(
            self.dropout(self_attn + x)
        )

        #Transformer block with src_mask to account for padding
        out = self.transformer_block(query, key, value, src_mask)

        return out

class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        num_layers,
        num_heads,
        forward_dim,
        dropout,
        max_len,
    ):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)

        # Positional encodings
        self.relative_positional_embedding = nn.Embedding(max_len, emb_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Decoder blocks
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    forward_dim=forward_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final linear layer: embedding_dim → vocab_size
        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        batch_size, seq_len = x.shape

        #Embeddings
        x_embed = self.token_embedding(x)

        #Positional indices
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(batch_size, seq_len)
        positions = positions.to(x.device)

        #Positional embeddings
        pos_embed = self.relative_positional_embedding(positions)

        #Combine token + positional embeddings
        out = x_embed + pos_embed

        #Dropout
        out = self.dropout(out)

        #Apply decoder blocks
        for layer in self.layers:
            out = layer(
                x=out,
                value=encoder_out,
                key=encoder_out,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
            )

        #Final projection to vocabulary logits
        out = self.output_layer(out)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        src_pad_idx,
        tgt_pad_idx,
        emb_dim=512,
        num_layers=6,
        num_heads=8,
        forward_dim=2048,
        dropout=0.0,
        max_len=128,
    ):
        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # Encoder
        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
        )

        # Decoder
        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            emb_dim=emb_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            forward_dim=forward_dim,
            dropout=dropout,
            max_len=max_len,
        )

    def create_src_mask(self, src):
        device = src.device
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(device)

    def create_tgt_mask(self, tgt):
        device = tgt.device
        batch_size, tgt_len = tgt.shape
        tgt_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=device)
        ).expand(batch_size, 1, tgt_len, tgt_len)
        tgt_mask = tgt_mask * causal_mask
        return tgt_mask.to(device)

    def forward(self, src, tgt):
        # Create masks
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        # Encode source sequence
        enc_out = self.encoder(src, src_mask)

        # Decode target sequence
        out = self.decoder(tgt, enc_out, src_mask, tgt_mask)

        return out