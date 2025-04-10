# RAG-Based Tweet Authenticity Classifier

This project implements a Retrieval-Augmented Generation (RAG) based approach to classify tweets as "real" or "fake". It leverages a retriever model (Sentence Transformer) to find relevant examples from the training data and incorporates them as context for a final classifier model (e.g., BERT, RoBERTa).

## Overview

The core idea is to enhance the classification performance by providing the classifier not just with the tweet itself, but also with similar tweets from a known dataset (the training set in this case). This context can help the model make more informed decisions, especially for nuanced or ambiguous cases.

The workflow involves:
1.  **Retriever Indexing:** Encoding all training tweets using a Sentence Transformer model and building a FAISS index for efficient similarity search.
2.  **Retrieval:** For each input tweet (during training, validation, or testing), retrieve the top-k most similar tweets from the FAISS index.
3.  **Augmentation & Classification:** Combine the original tweet and the retrieved tweets into a single input sequence, then feed it into a transformer-based sequence classification model for the final prediction (real/fake).

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