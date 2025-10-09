# Browser Environment Module

Playwright-based browser automation environment for GUI agent interaction.

## Components

- **`envs.py`**: Main environment class (`ScriptBrowserEnv`)
  - Gymnasium-compatible interface
  - Screenshot-based observations
  - Action execution and state management
  - Page lifecycle management

- **`actions.py`**: Action type definitions and enumerations

- **`action_parser_ground.py`**: Action parsing and execution logic
  - Pixel-level action grounding
  - Element localization
  - Action validation

- **`processors.py`**: Observation processors
  - Image observation processing
  - Text observation processing
  - Screenshot capture and formatting

- **`helper_functions.py`**: Utility functions for rendering and visualization

- **`utils.py`**: Common utilities and type definitions

- **`auto_login.py`**: Automated login handling for websites

- **`constants.py`**: Environment constants and configurations

- **`trajectory.py`**: Trajectory data structures

## Usage

```python
from browser_env import ScriptBrowserEnv

env = ScriptBrowserEnv(
    headless=True,
    viewport_size={"width": 1280, "height": 720},
    save_trace_enabled=True
)

# Reset environment with config
obs, info = env.reset(options={"config_file": "config.json"})

# Execute action
obs, reward, done, truncated, info, url = env.step(action, observation=obs)

# Cleanup
env.close()
```
