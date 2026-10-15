# Multi-Task Transformer & Causal RAG Engine

A PyTorch implementation from scratch of a Multi-Task Transformer Encoder and Retrieval-Augmented Generation (RAG) Causal Decoder for e-commerce review analysis.

## Overview

Designed without high-level pre-trained wrapper libraries, this project implements core transformer mechanics in native PyTorch: multi-head self-attention, positional encoding, multi-task prediction heads, and causal masked generation. The pipeline processes user reviews, classifies sentiment alongside purchase veracity, and generates context-aware explanations using cosine retrieval over learned embedding representations.

## Architecture

### 1. Multi-Task Transformer Encoder
- Native PyTorch encoder with scaled dot-product multi-head attention and feed-forward sublayers.
- Multi-task output heads via `[CLS]` token pooling:
  - Sentiment classification: Negative, Neutral, Positive.
  - Purchase verification: Verified vs. unverified transactions.
- Joint training using a weighted multi-task cross-entropy loss.

### 2. Custom Tokenization Pipeline
- Zero-dependency vocabulary generation and tokenization tailored for review text.
- Manages sequence padding, truncation, dynamic batch collation, and special token injection (`<BOS>`, `<EOS>`, `<CLS>`, `<PAD>`, `<UNK>`).

### 3. Retrieval-Augmented Generation Pipeline
- **Vector Indexing**: Extracts `[CLS]` embedding representations across the review corpus.
- **Context Retrieval**: Computes cosine similarity to select the top-k relevant reviews for input queries.
- **Causal Transformer Decoder**: Auto-regressive decoder with causal triangular masking to synthesize retrieved review contexts into human-readable explanations.

## Tech Stack

- PyTorch
- NumPy, Pandas, Scikit-learn
- Python 3.10+

## Pipeline Flow

1. **Preprocessing**: Raw JSON review dumps are cleaned, tokenized, and mapped to integer sequences.
2. **Phase 1 (Representation & Classification)**: The encoder learns contextual representations while optimizing joint classification objectives.
3. **Phase 2 (Index Building)**: Normalized encoder representations populate a semantic search vector index.
4. **Phase 3 (Retrieval & Generation)**: Query embeddings retrieve top-k context passages, and the causal decoder generates explanatory text conditioned on the retrieved reviews.

## Repository Structure

- `ecommerce_multitask_rag_engine.ipynb`: Self-contained notebook containing data loading, model architectures, training loops, evaluation metrics, and inference demonstrations.
- `models/`: Local target directory for saved `.pt` model checkpoints (git-ignored).
- `results/`: Local target directory for training loss curves, evaluation reports, and confusion matrices (git-ignored).

---

Built by Aimal Khan.
