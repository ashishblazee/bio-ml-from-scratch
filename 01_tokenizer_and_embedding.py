""" 01_tokenizer_and_embeddings.py
=====================================
Milestone 1: Foundations of protein deep learning from scratch
Building an Amino acid Tokenizer and learnable vector embedding layer in PyTorch

Author: Ashish(github.com/ashishblazee)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#1. The protein tokenizer

class ProteinTokenizer:
    def __init__(self):
        self.special_tokens = ["<PAD>" , "<UNK>" , "<BOS>" , "<EOS>" , "<MASK>"]
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        self.vocab = self.special_tokens + self.amino_acids
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        self.pad_id = self.char_to_idx["<PAD>"]
        self.unk_id = self.char_to_idx["<UNK>"]
        self.bos_id = self.char_to_idx["<BOS>"]
        self.eos_id = self.char_to_idx["<EOS>"]
        self.mask_id = self.char_to_idx["<MASK>"]

    def encode(self, sequence: str) -> torch.Tensor:
        """Converts a single sequence string(e.g "HVE) into a 1D Tensor of token IDs"""
        tokens = [self.bos_id]
        for aa in sequence.upper():
            tokens.append(self.char_to_idx.get(aa, self.unk_id))
        tokens.append(self.eos_id)
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        """converts a 1D tensor of token IDs back into a readable string."""
        return "".join([self.idx_to_char[int(i)] for i in token_ids])

    def batch_encode(self, sequences: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Takes a list of sequences with diffrent lengths, adds <BOS>/<EOS>,
        and pads them with <PAD> so they form a rectangular 2D tensor
        
        returns:
            padded_tokens: Tensor of shape (Batch_Size, Max_Length)
            attention_mask: Tensor of shape (Batch_size , Max_length) (1 for real residue, 0 for PAD)
        """
        encoded_list = [self.encode(seq) for seq in sequences]
        max_len = max(len(t) for t in encoded_list)

        batch_size = len(sequences)
        padded_tokens = torch.full((batch_size, max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

        for i, tokens in enumerate(encoded_list):
            padded_tokens[i, :len(tokens)] = tokens
            attention_mask[i, :len(tokens)] = 1 # 1 = real residues, 0 = PAD

        return padded_tokens, attention_mask

    #2. Protein Embedding Layer

class ProteinEmbedding(nn.Module):
    """Transform integers token IDs into continuosus , learned vector embeddings.
    convert tensor (Batch, Length) -> Tensor (Batch, length, Embedding_Dim).
    """
    def __init__(self, vocab_size: int = 25, embedding_dim: int = 64, max_seq_len: int = 512):
        super().__init__()
        #1. maps each amino acid to a vector
        self.token_embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = 0
        )

        #2. Learnable postional embedding(teaches the model the residue's order along the chain)
        self.position_embedding = nn.Embedding(
            num_embeddings = max_seq_len,
            embedding_dim = embedding_dim
        )

        #3. Layer Normalization for numerical stability
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        #Generate position indices: [0, 1, 2, ....., seq_len-1]
        positions = torch.arange(seq_len, device = input_ids.device).unsqueeze(0).expand(batch_size,seq_len)

        #comine token identity + spatial position along the backbone
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.layer_norm(x)

    #3 Demo & verification run

if __name__ == "__main__":
    print("=" * 65)
    print("DEMO: Protein Tokenizer & Vector embedding pipeline")
    print("=" * 65)

    tokenizer = ProteinTokenizer()
    print(f"Vocabulary Size: {len(tokenizer.vocab)} tokens")
    print(f"Vocabulary: {tokenizer.vocab}\n")

    proteins = [
        "HVEGSFTADIASYIE",
        "DEVEVTLPENFVEIFAE"
    ]
    print(f"Input Sequences (Batch of {len(proteins)}):")
    for i, p in enumerate(proteins):
        print(f" Protein{i+1} ({len(p)} aa): {p}")

    #1. Batch Tokenization & Padding
    input_ids, attention_mask = tokenizer.batch_encode(proteins)
    print(f"\n1. Tokenized Tensor Shape: {input_ids.shape} --> (Batch Size, Max Length)")
    print("Padded Input IDs:\n", input_ids)
    print("Attention Mask (0 indicates <PAD>):\n", attention_mask.int())

    #2. Pass into Embedding Layer
    EMBEDDING_DIM = 64
    embedder = ProteinEmbedding(vocab_size=len(tokenizer.vocab), embedding_dim=EMBEDDING_DIM)

    embedded_vectors = embedder(input_ids)
    print(f"\n2. Embedded Tensor Shape: {embedded_vectors.shape} --> (Batch, Length, Dimensions")
    print(f" Each amino acid is now represented as a vector of {EMBEDDING_DIM} numbers!")

    #3. vector cosin similarity check(biochemical Intuition
    with torch.no_grad():
        leu_vec = embedder.token_embedding.weight[tokenizer.char_to_idx["L"]]
        ile_vec = embedder.token_embedding.weight[tokenizer.char_to_idx["I"]]
        lys_vec = embedder.token_embedding.weight[tokenizer.char_to_idx["K"]]

        sim_leu_ile = F.cosine_similarity(leu_vec, ile_vec, dim=0).item()
        sim_leu_lys = F.cosine_similarity(leu_vec, lys_vec, dim=0).item()

    print("f\n3. Initial vector cosine Similarities (Untrained):")
    print(f" Cosine Sim(Leu vs Ile - both hydrophobic): {sim_leu_ile:.4f}")
    print(f" Cosine Sim(Leu vs Lys - hydrophobic vs basic): {sim_leu_lys:.4f}")
    print(" (As a model trains, similar amino acids naturally learns higher similarity!!")
    print("=" * 65)










