# Configuration Module

Centralized argument parsing and configuration management for the GUI Agent.

## Components

- **`argument_parser.py`**: Command-line argument parser with all configuration options

## Configuration Categories

### Browser Environment
- `--viewport_width/height`: Browser viewport dimensions (default: 1280x720)
- `--headless`: Run browser in headless mode
- `--slow_mo`: Slow down actions for debugging
- `--save_trace_enabled`: Enable Playwright trace recording

### Agent Settings
- `--agent_type`: Agent type (default: "prompt")
- `--max_steps`: Maximum steps per task (default: 10)
- `--max_obs_length`: Context length for observations
- `--model`: Model selection (qwen2.5-vl, ui-tars, etc.)

### Memory & Experience
- `--use_memory`: Enable experience memory
- `--use_continuous_memory`: Use continuous memory with transformers
- `--memory_retrieval_topk`: Number of memories to retrieve

### Evaluation
- `--evaluation_type`: Type of evaluation (mmina, webvoyager, mind2web, etc.)
- `--domain`: Domain for evaluation
- `--test_start_idx/test_end_idx`: Test range
- `--result_dir`: Output directory for results

### Data Collection
- `--collect_training_data`: Enable training data collection
- `--training_data_dir`: Directory for training data

## Usage

```python
from config.argument_parser import config

args = config()
print(f"Model: {args.model}")
print(f"Max steps: {args.max_steps}")
```

Or via command line:
```bash
python run.py --model qwen2.5-vl --max_steps 15 --domain shopping
```

