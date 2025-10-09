# Memory Module

Experience memory system for storing and retrieving agent trajectories.

## Components

- **`experience_memory.py`**: Core memory management class
  - FAISS-based vector indexing
  - Multimodal similarity search (text + images)
  - Experience storage and retrieval
  - Trajectory filtering and ranking

- **`help_functions.py`**: Helper functions for memory operations
  - CLIP-based text similarity
  - CLIP-based multimodal similarity
  - Embedding generation

## Features

### Experience Storage
- Store trajectories with actions and screenshots
- Metadata tracking (task descriptions, URLs, success)
- JSON-based persistence

### Similarity Search
- Text-only similarity using CLIP text encoder
- Multimodal similarity with images
- Configurable top-k retrieval

### Memory Indexing
- FAISS index for fast similarity search
- Support for both flat and IVF indices
- Automatic index saving/loading

## Usage

```python
from memory.experience_memory import Memory

# Initialize memory
memory = Memory(
    training_data_path="training_data/",
    faiss_index_path="memory/memory_index/",
    multimodal=True
)

# Retrieve relevant experiences
results = memory.retrieve(
    query="Search for hotels in Tokyo",
    top_k=5
)

# Process retrieved experiences
for trajectory, score in results:
    print(f"Score: {score}")
    print(f"Actions: {trajectory['actions']}")
```

## Index Directory Structure

```
memory_index/
├── index.faiss          # FAISS index file
├── metadata.json        # Trajectory metadata
└── config.json          # Index configuration
```

## Notes

- Requires CLIP model for embedding generation
- FAISS indices are built on first use
- Supports incremental updates to memory

