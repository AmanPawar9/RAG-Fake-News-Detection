# RAG-Based Tweet Authenticity Classifier

This project implements a Retrieval-Augmented Generation (RAG) based approach to classify tweets as "real" or "fake". It leverages a retriever model (Sentence Transformer) to find relevant examples from the training data and incorporates them as context for a final classifier model (e.g., BERT, RoBERTa).

## Overview

The core idea is to enhance the classification performance by providing the classifier not just with the tweet itself, but also with similar tweets from a known dataset (the training set in this case). This context can help the model make more informed decisions, especially for nuanced or ambiguous cases.

The workflow involves:
1.  **Retriever Indexing:** Encoding all training tweets using a Sentence Transformer model and building a FAISS index for efficient similarity search.
2.  **Retrieval:** For each input tweet (during training, validation, or testing), retrieve the top-k most similar tweets from the FAISS index.
3.  **Augmentation & Classification:** Combine the original tweet and the retrieved tweets into a single input sequence, then feed it into a transformer-based sequence classification model for the final prediction (real/fake).


# Graph-RAG-Based Tweet Authenticity Classifier
This project implements a Graph-Retrieval-Augmented Generation (Graph-RAG) based approach to classify tweets as "real" or "fake". It leverages a retriever model (Sentence Transformer) to find relevant examples from the training data using a neighborhood exploration strategy and incorporates them as context for a final classifier model (e.g., BERT, RoBERTa).

# Overview
The core idea is to enhance the classification performance by providing the classifier not just with the tweet itself, but also with semantically similar tweets and their neighbors from a known dataset (the training set in this case). This simulates a 1-hop graph traversal in the embedding space, potentially capturing richer contextual relationships than standard RAG.

The workflow involves:
1. **Retriever Indexing:** Encoding all training tweets using a Sentence Transformer model and building a FAISS index for efficient similarity search.
2. **Graph-like Retrieval:** For each input tweet (during training, validation, or testing):
3. **Retrieve the top-k:** primary most similar tweets from the FAISS index.
4. **Retrieve the top-m:** secondary neighbors for each of those primary tweets.
5. **Combine and deduplicate:** these primary and secondary neighbors.

Augmentation & Classification: Combine the original tweet and the final set of retrieved neighbor tweets into a single input sequence, then feed it into a transformer-based sequence classification model for the final prediction (real/fake).

# Fast Graph-RAG Implementation (Adapted for Nano/Speed).

This script implements a Retrieval-Augmented Generation (RAG) approach
for text classification, adapted for faster execution ("Fast-Graph-RAG" or "Nano-Graph-RAG").
Key adaptations include:
- Using DistilBERT as the classifier for faster training/inference.
- Employing FAISS IndexIVFFlat for faster approximate nearest neighbor search.
- Adding optional mixed-precision training (torch.cuda.amp).
- Configuration options tailored for speed vs. accuracy trade-offs.

Core Components:
1. **Configuration:** Hyperparameters, paths, model names, FAISS settings, AMP flag.
2. **Data Loading:** Loads and preprocesses data.
3. **Custom Dataset (GraphRagNewsDataset):** Integrates retrieval.
4. **Retriever Setup:** SentenceTransformer + FAISS (IndexIVFFlat).
5. **Graph-like Retrieval Function:** Finds primary/secondary neighbors using FAISS.
6. **Classifier Model:** DistilBERT for sequence classification.
7. **Training Loop:** Fine-tunes the classifier with optional AMP.
8. **Evaluation:** Measures performance.

## Features

* **RAG Implementation:** Combines retrieval and classification for enhanced performance.
* **Efficient Retrieval:** Uses `sentence-transformers` for dense embeddings and `faiss` for fast nearest neighbor search.
* **Flexible Classifier:** Uses `transformers` library, allowing easy swapping of models like BERT, RoBERTa, etc.
* **Standard Training Pipeline:** Includes training, validation (with early stopping based on F1-score), and final testing phases.
* **Configuration Class:** Centralized configuration for file paths, model names, hyperparameters, and device settings.
* **Data Loading & Preprocessing:** Handles loading data from Excel files, cleaning, and label mapping.
* **Custom PyTorch Dataset:** `RagNewsDataset` seamlessly integrates the retrieval step into the data loading process.
* **Multiprocessing Aware:** Includes settings for `torch.multiprocessing` ('spawn' start method) and handles potential tokenizer parallelism warnings.
* **Debugging Support:** Includes a `DEBUG_DATALOADER` flag to simplify debugging data loading issues by forcing single-process loading.
* **Detailed Error Handling:** Incorporates `try-except` blocks and `traceback` printing for easier troubleshooting.

## Requirements

* Python 3.8+
* PyTorch (>= 1.8, check compatibility with your CUDA version if using GPU)
* Transformers
* Sentence Transformers
* Faiss (CPU or GPU version)
* Pandas
* NumPy
* Scikit-learn
* TQDM
* Openpyxl (for reading Excel files)

You can install the necessary packages using pip:

```bash
pip install torch pandas numpy scikit-learn tqdm openpyxl transformers sentence-transformers
# Install Faiss (choose one):
# For CPU:
pip install faiss-cpu
# For GPU (requires CUDA toolkit installed):
# Check Faiss documentation for specific CUDA version compatibility
pip install faiss-gpu
```