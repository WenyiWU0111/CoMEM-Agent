# Data Preparation Module

Scripts for preparing training and inference data with memory embeddings.

## Components

- **`prepare_inference_data_memory.py`**: Prepare inference data with memory retrieval
  - Index training trajectories using FAISS
  - Generate embeddings for tasks
  - Retrieve relevant experiences for new tasks

- **`prepare_training_data_onfly.py`**: Real-time training data collection during evaluation

- **`help_functions.py`**: Utility functions for data processing

## Features

### Memory Indexing
- Build FAISS index from training trajectories
- Support multimodal embeddings (text + images)
- Efficient similarity search

### Experience Retrieval
- Query-based memory retrieval
- Top-k similar trajectory retrieval
- Task-specific experience filtering

## Usage

### Prepare Memory Embeddings

```python
from data_preparation.prepare_inference_data_memory import get_inference_memory_embeddings

# Index training data and retrieve for evaluation
get_inference_memory_embeddings(
    trajectory_path="path/to/training/data",
    dataset="mmina",
    domain="shopping"
)
```

### Build FAISS Index

The module automatically builds and saves FAISS indices for efficient memory retrieval during inference.

## Data Format

Training data should include:
- Task descriptions (intent)
- Action sequences
- Screenshots at each step
- Success indicators

