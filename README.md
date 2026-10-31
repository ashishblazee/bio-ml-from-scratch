# bio-ml-from-scratch

A modular, first-principles implementation of protein language models and transformer architectures in PyTorch.

Instead of relying on high-level pre-trained wrappers, this repository builds the core computational machinery of modern protein models (such as ESM-2 and AlphaFold) from scratch: tokenization, learnable structural embeddings, scaled dot-product attention, multi-head interaction maps, and pre-LayerNorm transformer blocks.

---

## Architecture Flow

```
Polypeptide Sequence ("ACDEF...")
       │
       ▼
[01] Tokenizer & Embedding Layer ──► Token IDs + Backbone Positional Vectors (d_model=64)
       │
       ▼
[02] Multi-Head Self-Attention   ──► Inter-residue contact maps (Q, K, V across 4 heads)
       │
       ▼
[03] Pre-LN Transformer Block    ──► Pre-LN + Attention + GELU FFN (d_ff=256) + Skip Connections
       │
       ▼
[04] Full ESM-2 Model            ──► Stacked N layers + Masked Language Modeling (MLM) Head
```

---

## Roadmap & Milestones

### Milestone 1: Amino Acid Tokenization & Learnable Embeddings
*File: `01_tokenizer_and_embedding.py`*

- **Vocabulary**: 20 canonical amino acids + 5 special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>`).
- **Batch Encoding**: Dynamic sequence padding and rectangular attention mask generation (`1` for real residues, `0` for `<PAD>`).
- **Dual Embedding Layer**:
  - `token_embedding`: Maps discrete amino acid IDs into dense continuous vector space ($\mathbb{R}^{d_{\text{model}}}$).
  - `position_embedding`: Learnable positional embeddings along the linear polypeptide chain (N-terminus $\to$ C-terminus).
- **Biochemical Cosine Similarity**: Evaluates vector representations between biochemically similar residues (e.g. Leu vs. Ile) versus discordant residues (e.g. Leu vs. Lys).

---

### Milestone 2: Scaled Dot-Product & Multi-Head Self-Attention
*File: `02_self_attention.py`*

- **Biological Intuition**: Sequence-distant residues interact in 3D space through disulfide bonds, salt bridges, and catalytic sites. Self-attention captures these long-range dependencies.
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - Attention masking ensures real residues pay 0% attention to padding tokens.
  - Softmax normalization over `dim=-1` enforces an exact 100% attention budget per residue.
- **Multi-Head Self-Attention (MHSA)**:
  - Splits $d_{\text{model}} = 64$ across 4 parallel heads ($d_{\text{head}} = 16$).
  - Reshapes tensors to `[Batch, Heads, Seq_Len, d_head]` to compute attention heads in parallel.
  - Output attention shape: `[Batch, 4, Seq_Len, Seq_Len]`, producing 4 simultaneous residue-residue interaction maps.

---

### Milestone 3: ESM-2 Style Pre-LN Transformer Block
*File: `03_transformer_block.py`*

- **Two Core Operations**:
  1. **Inter-residue Communication**: Multi-Head Self-Attention mixes context across positions in the protein.
  2. **Intra-residue Chemical Computation**: Position-wise Feed-Forward Network (FFN) with $4\times$ hidden expansion ($64 \to 256 \to 64$) and smooth GELU activations.
- **Pre-LayerNorm Architecture (Pre-LN)**:
  $$x_1 = x + \text{Dropout}(\text{Attention}(\text{LayerNorm}_1(x)))$$
  $$x_2 = x_1 + \text{Dropout}(\text{FFN}(\text{LayerNorm}_2(x_1)))$$
  - Standard in modern foundation models (ESM-2, AlphaFold, GPT).
  - Preserves identity gradient flow through the residual highway, stabilizing activations ($\mu \approx 0.0, \sigma \approx 1.0$) across deep architectures.

---

### Milestone 4: ESM-2 Model & Masked Language Modeling (MLM)
*File: `04_esm_model.py` (In Progress)*

- Stacking $N$ Pre-LN Transformer blocks into a deep encoder.
- Masked Language Modeling (MLM) prediction head: 15% random residue masking (`<MASK>`) to train the model to predict evolutionary sequence distributions from structural context.

---

### Milestone 5: Contact Map Extraction & Zero-Shot Variant Effect Prediction
- Extracting pairwise attention maps to predict 3D $\text{C}_\beta - \text{C}_\beta$ contact maps (< 8 Å).
- Zero-shot scoring of pathogenic vs. benign missense mutations using sequence log-odds:
  $$\Delta \text{score} = \log P(\text{mutant}) - \log P(\text{wildtype})$$

---

## Getting Started

### Requirements
- Python 3.10+
- PyTorch 2.0+

```bash
git clone https://github.com/ashishblazee/bio-ml-from-scratch.git
cd bio-ml-from-scratch
pip install torch
```

### Running the Milestones

**Milestone 1 (Tokenizer & Embeddings):**
```bash
python 01_tokenizer_and_embedding.py
```

**Milestone 2 (Multi-Head Self-Attention):**
```bash
python 02_self_attention.py
```

**Milestone 3 (Pre-LN Transformer Block):**
```bash
python 03_transformer_block.py
```

---

## Repository Structure

```
bio-ml-from-scratch/
├── LICENSE                       # MIT License
├── README.md                     # Project documentation and roadmap
├── .gitignore                    # Python & IDE exclusion rules
├── 01_tokenizer_and_embedding.py # Milestone 1: Tokenizer & learnable embeddings
├── 02_self_attention.py          # Milestone 2: Scaled Dot-Product & Multi-Head Attention
└── 03_transformer_block.py       # Milestone 3: Pre-LN ESM-2 Transformer Encoder Block
```

---

## Author

**Ashish**  
GitHub: [@ashishblazee](https://github.com/ashishblazee)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
