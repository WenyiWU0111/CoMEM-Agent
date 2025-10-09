# Utils Module

Shared utility functions and helper classes used across the project.

## Components

- **`help_functions.py`**: Core helper functions
  - Test file list creation for different benchmarks
  - Result tracking and unfinished task detection
  - Configuration dumping
  - Global variable management

- **`logging_setup.py`**: Logging configuration
  - Logger initialization
  - Log file management
  - Timestamp handling

- **`llm_wrapper.py`**: LLM wrapper classes
  - Unified interface for different LLM APIs
  - Token usage tracking
  - Error handling

- **`early_stop.py`**: Early stopping logic
  - Parsing failure detection
  - Repeating action detection
  - Maximum step enforcement

- **`action_check.py`**: Action validation utilities
  - Action format verification
  - Coordinate validation
  - Element ID checking

- **`training_data_collector.py`**: Training data collection
  - Conversation tracking
  - Prompt-response pairs logging
  - Trajectory saving



## Usage Examples

### Get Unfinished Tasks
```python
from utils.help_functions import get_unfinished

unfinished = get_unfinished(
    config_files=all_configs,
    result_dir="results/",
    config_dir="mmina"
)
```

### Check Action Validity
```python
from utils.action_check import validate_action

is_valid, error = validate_action(action)
if not is_valid:
    logger.error(f"Invalid action: {error}")
```

