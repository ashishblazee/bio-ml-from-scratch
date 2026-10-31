# Bio-ML From Scratch: Protein Language Models in PyTorch

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

> Building modern Protein Language Models (ESM-2, ProtBERT, AlphaFold-style architectures) from mathematical, chemical, and biological first principles using pure PyTorch.

---

## 🧬 Project Overview

State-of-the-art biological foundation models like **ESM-2** (Evolutionary Scale Modeling) and **AlphaFold** have revolutionized structural biology and de novo protein design. However, using high-level pre-trained wrappers obscures how transformers actually process the language of life.

**`bio-ml-from-scratch`** is a ground-up implementation of protein deep learning architectures built residue-by-residue, matrix-by-matrix, without relying on black-box model libraries.

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

## 🗺️ Roadmap & Milestones

### ✅ Milestone 1: Amino Acid Tokenization & Learnable Embeddings
*File: `01_tokenizer_and_embedding.py`*

- **Vocabulary**: 20 canonical amino acids + 5 special tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, `<MASK>`).
- **Batch Encoding**: Dynamic sequence padding and rectangular attention mask generation (`1` for real residues, `0` for `<PAD>`).
- **Dual Embedding Layer**:
  - `token_embedding`: Maps discrete amino acid IDs into dense continuous vector space ($\mathbb{R}^{d_{\text{model}}}$).
  - `position_embedding`: Learnable positional embeddings along the linear polypeptide chain (N-terminus $\to$ C-terminus).
- **Biochemical Cosine Similarity**: Evaluates untrained vector distance between biochemically similar residues (e.g. Leu vs. Ile) vs. discordant residues (e.g. Leu vs. Lys).

---

### ✅ Milestone 2: Scaled Dot-Product & Multi-Head Self-Attention
*File: `02_self_attention.py`*

- **Biological Intuition**: Proteins are non-linear 3D objects; sequence-distant residues interact through space (disulfide bonds, salt bridges, active site catalytic triads).
- **Scaled Dot-Product Attention**:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  - Includes attention masking to ensure real residues pay 0% attention budget to padding tokens.
  - Softmax normalized across `dim=-1` so every residue allocates exactly 100% attention budget across the sequence.
- **Multi-Head Self-Attention (MHSA)**:
  - Divides $d_{\text{model}} = 64$ across 4 parallel heads ($d_{\text{head}} = 16$).
  - Reshapes tensors to `[Batch, Heads, Seq_Len, d_head]` so PyTorch treats each head as an independent parallel contact lens.
  - Output attention tensor shape: `[Batch, 4, Seq_Len, Seq_Len]` — producing 4 simultaneous pairwise residue interaction contact maps.

---

### ✅ Milestone 3: ESM-2 Style Pre-LN Transformer Block
*File: `03_transformer_block.py`*

- **Two Core Pillars**:
  1. **Inter-residue Communication**: Multi-Head Self-Attention mixes context across all positions in the protein.
  2. **Intra-residue Chemical Computation**: Position-wise Feed-Forward Network (FFN) expands each residue $4\times$ ($64 \to 256 \to 64$) with smooth **GELU** activations to compute non-linear chemical features.
- **Pre-LayerNorm Architecture (Pre-LN)**:
  $$x_1 = x + \text{Dropout}(\text{Attention}(\text{LayerNorm}_1(x)))$$
  $$x_2 = x_1 + \text{Dropout}(\text{FFN}(\text{LayerNorm}_2(x_1)))$$
  - Modern architecture used by ESM-2, AlphaFold, and GPT.
  - Leaves the identity residual stream clean, preventing vanishing/exploding gradients and keeping output activations stable ($\mu \approx 0.0, \sigma \approx 1.0$) across deep 30+ layer networks.

---

### ⏳ Milestone 4: ESM-2 Model Architecture & Masked Language Modeling (MLM)
*File: `04_esm_model.py` (In Progress)*

- Stacking $N$ Pre-LN Transformer blocks into a deep encoder.
- Masked Language Modeling (MLM) prediction head: 15% random residue masking (`<MASK>`) to train the model to predict evolutionary sequence distributions from structural context.

---

### ⏳ Milestone 5: Contact Map Extraction & Zero-Shot Variant Effect Prediction
- Extracting pairwise attention maps to predict 3D $\text{C}_\beta - \text{C}_\beta$ contact maps (< 8 Å).
- Zero-shot scoring of pathogenic vs. benign missense mutations using sequence log-odds:
  $$\Delta \text{score} = \log P(\text{mutant}) - \log P(\text{wildtype})$$

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+

```bash
git clone https://github.com/ashishblazee/bio-ml-from-scratch.git
cd bio-ml-from-scratch
pip install torch
```

### Running the Milestones

**Test Milestone 1 (Tokenizer & Embeddings):**
```bash
python 01_tokenizer_and_embedding.py
```

**Test Milestone 2 (Multi-Head Self-Attention):**
```bash
python 02_self_attention.py
```

**Test Milestone 3 (Pre-LN Transformer Block):**
```bash
python 03_transformer_block.py
```

---

## 🔬 Directory Structure

```
bio-ml-from-scratch/
├── .gitignore                    # Python & IDE exclusion rules
├── README.md                     # Architecture documentation and roadmap
├── 01_tokenizer_and_embedding.py # Milestone 1: Tokenizer & learnable embeddings
├── 02_self_attention.py          # Milestone 2: Scaled Dot-Product & Multi-Head Attention
└── 03_transformer_block.py       # Milestone 3: Pre-LN ESM-2 Transformer Encoder Block
```

---

## 👨‍💻 Author

**Ashish**  
- GitHub: [@ashishblazee](https://github.com/ashishblazee)  
- Focus: Computational Biology, Deep Learning for Protein Design & Structure Prediction.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
