# End-to-End E-Commerce RAG & Multi-Task Analysis Engine

## Overview
This repository contains a custom-built, from-scratch implementation of a **Transformer-based Retrieval-Augmented Generation (RAG) system** in PyTorch. 

Designed without reliance on high-level pre-trained libraries (like HuggingFace `transformers`), this project showcases deep understanding of attention mechanisms, multi-task learning, and vector-based retrieval. The system is built to process raw e-commerce reviews, classify them across multiple objectives, and autonomously generate context-aware, explainable responses using a custom decoder.

## Key Features & Architecture

### 1. Multi-Task Transformer Encoder (From Scratch)
- Built a native PyTorch Encoder implementing Multi-Head Self-Attention, Positional Encoding, and Feed-Forward networks.
- **Multi-Task Learning:** The encoder utilizes a `[CLS]` token to simultaneously predict:
  - **Sentiment:** (Negative, Neutral, Positive)
  - **Purchase Verification:** Distinguishing verified vs. unverified purchases.
- Trained jointly using a weighted composite loss function.

### 2. Custom Data Pipeline & Tokenization
- Engineered a lightweight, zero-dependency text preprocessing pipeline.
- Implemented a custom byte-level/word-level vocabulary builder and tokenization engine tailored for messy user-generated e-commerce data.
- Handles padding, dynamic sequence lengths, and special token injection (`<BOS>`, `<EOS>`, `<CLS>`, `<PAD>`, `<UNK>`).

### 3. Retrieval-Augmented Generation (RAG) Pipeline
- **Vector Indexing:** Extracts and indexes high-dimensional `[CLS]` embeddings from the trained Encoder for the entire corpus.
- **Context Retrieval:** Uses cosine similarity to fetch the top-k most relevant historical reviews when queried.
- **Causal Transformer Decoder:** A from-scratch auto-regressive Decoder model featuring causal masking. It intakes the user query along with the retrieved context to generate coherent, explainable text regarding the product's sentiment.

## Tech Stack
- **Frameworks:** PyTorch
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Python 3.10+

## How it Works
1. **Preprocessing:** Raw `.json.gz` dumps are cleaned and tokenized.
2. **Phase 1 (Understanding):** The multi-task encoder learns deep contextual representations while classifying sentiment and veracity.
3. **Phase 2 (Retrieval):** The trained encoder maps the corpus into a searchable semantic vector space.
4. **Phase 3 (Generation):** A user query fetches relevant contexts from the vector space. The custom Causal Decoder synthesizes this context into a human-readable explanation.

## Project Structure
- `*.ipynb` - Core implementation containing the architectures, training loops, and evaluation matrices.
- `/models/` - Directory storing the `.pt` weights for both the Encoder and Decoder.
- `/results/` - Training logs, learning curves, and evaluation artifacts (F1-scores, Confusion Matrices).

---
*This project serves as a technical showcase for advanced Deep Learning engineering, specifically targeting core NLP architectures, custom training loops, and end-to-end RAG pipelines.*
