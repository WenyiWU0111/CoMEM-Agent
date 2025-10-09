# Agent Module

Core agent implementation with ReAct paradigm for GUI automation using vision-language models.

## Components

- **`agent.py`**: Main agent class implementing the ReAct (Reasoning + Acting) framework
  - Multimodal observation processing (screenshots + text)
  - Function calling with tools (click, type, scroll, web search, etc.)
  - Memory retrieval for experience-based learning
  - Action generation and validation

- **`llm_config.py`**: LLM configuration and model loaders
  - DirectVLLMModel: vLLM server wrapper for fast inference
  - DirectTransformersModel: Transformers-based model with experience handling
  - Model factory functions for different VLMs (Qwen2.5-VL, UI-TARS, etc.)

- **`prompts/`**: System prompts and examples for agent behavior
  - `system_prompt.txt`: Main instruction template
  - `system_prompt_simple.txt`: Simplified version
  - `examples.txt`: Few-shot examples

## Key Features

- **Vision-Language Understanding**: Process screenshots and understand GUI elements
- **Tool Use**: Leverage function calling for structured actions
- **Memory Integration**: Retrieve and use past experiences
- **Multi-step Reasoning**: Chain actions to complete complex tasks

## Agent Construction

```python
from agent import construct_agent
from config.argument_parser import config

args = config()
agent = construct_agent(args)

# Use agent to generate next action
action, metadata = agent.next_action_custom(trajectory, intent, meta_data)
```

