import torch 
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module  

MultiHeadSelfAttention = import_module("02_self_attention").MultiHeadSelfAttention

class PositionwiseFeedForward(nn.Module):
    """
    Two-layer MLP applied to each amino acid position separately.
    Expands d_model (64) -> d_ff (256) -> d_model (64).
    GELU is the smooth activation used by ESM-2 and BERT.
    """
    def __init__(self, d_model: int = 64, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x): 
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single ESM-2 style Transformer encoder layer 
    Pre-LayerNorm architecture:
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """ 
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Residue-to-residue communication (Self-Attention)
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attention(norm_x, mask=mask)
        x = x + self.dropout1(attn_out) # Residual connection!

        # 2. Individual residue chemical processing (FFN)
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout2(ffn_out) # Residual connection!

        return x, attn_weights

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: ESM-2 Style Transformer Block (Pre-LN)")
    print("=" * 60)

    # 1 sequence of 5 amino acid residues, each embedded with 64 features 
    dummy_protein = torch.randn(1, 5, 64)

    block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
    out, attn_weights = block(dummy_protein)

    print("Input tensor shape:      ", dummy_protein.shape)
    print("Output tensor shape:     ", out.shape)
    print("Attention weights shape: ", attn_weights.shape)
    print("\nResidual check:")
    print("Mean of output features: ", f"{out.mean().item():.4f}")
    print("Std dev of features:     ", f"{out.std().item():.4f}")
