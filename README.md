# RAG System using LLaMA and LlamaIndex

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![LLaMA](https://img.shields.io/badge/LLaMA-2%207B-orange?style=for-the-badge)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Framework-green?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge)

**A scalable Retrieval-Augmented Generation pipeline using Meta's LLaMA 2 for context-aware question answering**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Architecture](#architecture)

</div>

## Overview

This project implements a sophisticated RAG (Retrieval-Augmented Generation) system that combines Meta's LLaMA 2 (7B-chat) model with LlamaIndex for intelligent document retrieval and question answering. The system processes documents, creates vector embeddings, and provides accurate, context-aware responses to user queries.

## Features

- **Scalable RAG Pipeline** - Built with LlamaIndex for efficient document processing and retrieval
- **LLaMA 2 Integration** - Utilizes Meta's 7B-chat model with optimized inference settings
- **Advanced Embeddings** - Uses sentence-transformers/all-mpnet-base-v2 for high-quality vector representations
- **Memory Optimization** - 8-bit quantization and FP16 precision for efficient deployment
- **Document Processing** - Automatic chunking and indexing of various document formats
- **Context-Aware QA** - Retrieves relevant context before generating responses

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language Model** | Meta LLaMA 2 (7B-chat-hf) |
| **RAG Framework** | LlamaIndex |
| **ML Library** | HuggingFace Transformers |
| **Embeddings** | Sentence Transformers (all-mpnet-base-v2) |
| **Optimization** | 8-bit quantization, FP16 precision |
| **Backend** | PyTorch, CUDA |
| **Document Processing** | PyPDF, SimpleDirectoryReader |

## Prerequisites

- **Python 3.8+**
- **CUDA-compatible GPU** (8GB+ VRAM recommended)
- **HuggingFace Account** with access to LLaMA 2 models
- **16GB+ RAM** (for optimal performance)

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/rag-llama-system.git
cd rag-llama-system
```

### 2. Install Dependencies
```bash
pip install pypdf
pip install transformers einops accelerate langchain bitsandbytes
pip install sentence_transformers
pip install llama_index
```

### 3. HuggingFace Authentication
```bash
huggingface-cli login
```
Enter your HuggingFace token when prompted.

### 4. Prepare Documents
Create a `data/` directory and add your documents:
```bash
mkdir data
# Add your PDF, TXT, or other documents to the data/ folder
```

## Usage

### Basic Implementation

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import HuggingFaceLLM
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import torch

# Load documents
documents = SimpleDirectoryReader("/path/to/data").load_data()

# Configure LLaMA 2 model
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True}
)

# Create service context with embeddings
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Build index and query engine
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()

# Query the system
response = query_engine.query("What is attention mechanism?")
print(response)
```

### Sample Queries

```python
# Technical questions
response = query_engine.query("What is attention is all you need?")
print(response)

# Computer vision questions
response = query_engine.query("What is YOLO?")
print(response)

# Custom domain questions based on your documents
response = query_engine.query("Explain the main concepts in the document")
print(response)
```

## Architecture

```
Documents → Document Loader → Text Chunking → Vector Embeddings
                                                      ↓
User Query → Query Engine ← Vector Index ← Embedding Model
     ↓
Context Retrieval → LLaMA 2 Model → Generated Response
```

### Key Components

#### Document Processing
- **SimpleDirectoryReader** - Loads documents from specified directory
- **Text Chunking** - Splits documents into 1024-token chunks for optimal processing
- **Format Support** - PDF, TXT, and other common document formats

#### Embedding Generation
- **Model**: sentence-transformers/all-mpnet-base-v2
- **Vector Store**: LlamaIndex VectorStoreIndex
- **Similarity Search**: Cosine similarity for document retrieval

#### Language Model
- **Model**: Meta LLaMA 2 (7B-chat-hf)
- **Optimization**: 8-bit quantization, FP16 precision
- **Context Window**: 4096 tokens
- **Generation**: Temperature 0.0 for deterministic responses

## Configuration Options

### Model Parameters
```python
# Adjust generation settings
generate_kwargs = {
    "temperature": 0.1,      # Creativity vs accuracy
    "do_sample": True,       # Enable sampling
    "top_p": 0.9,           # Nucleus sampling
    "max_new_tokens": 512   # Response length
}

# Memory optimization
model_kwargs = {
    "torch_dtype": torch.float16,  # Half precision
    "load_in_8bit": True,         # 8-bit quantization
    "device_map": "auto"          # Automatic GPU allocation
}
```

### Service Context
```python
service_context = ServiceContext.from_defaults(
    chunk_size=1024,        # Document chunk size
    chunk_overlap=20,       # Overlap between chunks
    llm=llm,               # Language model
    embed_model=embed_model # Embedding model
)
```

## Performance Optimization

### Memory Usage
- **8-bit Quantization**: Reduces model size from ~13GB to ~7GB
- **FP16 Precision**: Further memory optimization
- **Chunked Processing**: Handles large documents efficiently

### Inference Speed
- **CUDA Acceleration**: GPU-optimized inference
- **Batch Processing**: Efficient document indexing
- **Caching**: Vector embeddings cached for reuse

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce chunk size or enable more aggressive quantization
   model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True}
   ```

2. **HuggingFace Authentication**
   ```bash
   # Re-authenticate if token expires
   huggingface-cli logout
   huggingface-cli login
   ```

3. **Document Loading Issues**
   ```python
   # Check document format and permissions
   documents = SimpleDirectoryReader("/content/data", required_exts=[".pdf", ".txt"]).load_data()
   ```

## Project Structure

```
rag-llama-system/
├── main.py                 # Main implementation
├── data/                   # Document directory
│   ├── document1.pdf
│   ├── document2.txt
│   └── ...
├── requirements.txt        # Dependencies
├── README.md              # This file
└── examples/              # Usage examples
    └── sample_queries.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/enhancement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/enhancement`)
5. Open a Pull Request


## Acknowledgments

- **Meta AI** for the LLaMA 2 model
- **LlamaIndex** for the RAG framework
- **HuggingFace** for model hosting and transformers library
- **Sentence Transformers** for embedding models

