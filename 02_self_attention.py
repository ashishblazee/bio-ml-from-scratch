import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # get the dimension size of the vectors (e.g. 64)
        d_k = q.size(-1)
        # compute raw compatibility scores between every residue pair
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        # mask out <PAD> positions so real residues pay 0% attention to padding 
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # convert raw scores into probabilities (0.0 to 1.0) using softmax 
        # dim=-1 ensures each row sums to exactly 1.0 (100% attention budget)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # weight the value vectors (v) by the attention probabilities
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # ensure the total dimensions divide evenly among the heads 
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # e.g. 64 // 4 = 16 numbers per head

        # linear projection matrices for Q, K, V 
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # our core single-head attention engine that we just built!!
        self.attention = ScaleDotProductAttention(dropout=dropout)  

        # final output projection to combine information from all heads 
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # project input into Q, K, V: shape (batch, seq_len, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # reshape into 4 heads: (batch, seq_len, 4, 16) -> (batch, 4, seq_len, 16)
        # we swap dimensions 1 and 2 so PyTorch treats each head as an independent batch!
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # adjust mask shape for multi-head broadcasting: (batch, 1, 1, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        # compute attention across all 4 heads in parallel!
        out, attn_weights = self.attention(q, k, v, mask=mask)

        # glue the 4 heads back together: (batch, 4, seq_len, 16) -> (batch, seq_len, 64)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # final linear projection 
        output = self.w_o(out)

        return output, attn_weights 


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Single-Head Scaled Dot-Product Attention")
    print("=" * 60)
    # create 3 dummy amino acids, each represented by 4 numbers
    # shape: (batch=1, length=3, dim=4)
    dummy_q = torch.randn(1, 3, 4)
    dummy_k = torch.randn(1, 3, 4)
    dummy_v = torch.randn(1, 3, 4)

    single_head = ScaleDotProductAttention()
    out, weights = single_head(dummy_q, dummy_k, dummy_v)

    print("Output vector shape:   ", out.shape)
    print("Attention weights shape:", weights.shape)
    print("\nAttention Matrix (3x3 grid between residues):")
    print(weights)
    print("\nRow sums (proves each row sums to 1.0!):")
    print(weights.sum(dim=-1))

    print("\n" + "=" * 60)
    print("TEST 2: Multi-Head Self-Attention (4 Heads, d_model=64)")
    print("=" * 60)
    # 1 sequence of 5 amino acid residues, each embedded as 64 features
    dummy_protein = torch.randn(1, 5, 64)

    mha = MultiHeadSelfAttention(d_model=64, num_heads=4)
    mha_out, mha_weights = mha(dummy_protein)

    print("Input protein tensor shape: ", dummy_protein.shape)
    print("Output protein tensor shape:", mha_out.shape)
    print("Multi-head attention maps:  ", mha_weights.shape)
    print("\nExplanation of [1, 4, 5, 5]:")
    print(" - 1: Batch size (1 protein)")
    print(" - 4: Attention heads (4 independent contact lenses!)")
    print(" - 5, 5: Pairwise 5x5 interaction grid between all 5 residues")
