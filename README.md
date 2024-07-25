# Fine-Tuning-and-Retrieval-Augmented-Generation-for-Telecom-Networks-
# Specializing Large Language Models for Telecom Networks

This project is part of the Zindi competition: [Specializing Large Language Models for Telecom Networks](https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks). The goal is to finetune the PHI-2 language model using telecom documents to improve its performance on telecom-specific tasks.

## Files and Modules

### `finetune.py`: Fine-Tuning PHI-2 Language Model
This script handles the fine-tuning of the PHI-2 language model on telecom-specific documents. Key functionalities include:
- Tokenizing input data.
- Preparing the model for k-bit training.
- Configuring the trainer and training arguments.

### `functions.py`: Utility Functions for Data Manipulation and Model Handling
This script contains various utility functions essential for data manipulation and model handling. Key functionalities include:
- Removing release numbers from data.
- Setting up the `RetrieverQueryEngine`.

### `main.py`: Main Execution Script for Language Model Operations
The main entry point for the project. It sets up and runs the PHI-2 language model with various configurations. Key functionalities include:
- Initializing the model and tokenizer.
- Setting up embeddings and vector stores using Llama Index.

### `rag.py`: Retrieval-Augmented Generation (RAG) Model Setup
This script focuses on setting up the retrieval-augmented generation (RAG) model. Key functionalities include:
- Initializing embeddings and vector stores with Llama Index and Chroma.
- Setting up the `RetrieverQueryEngine`.

## Installation

To install the required packages, run:
```bash
pip install -r requirements.txt
