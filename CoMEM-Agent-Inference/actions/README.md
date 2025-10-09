# Actions Module

This module provides action creation and parsing utilities for GUI interactions.

## Components

- **`action_creator.py`**: Factory functions to create different types of browser actions:
  - Click actions (element interaction)
  - Type actions (text input)
  - Scroll actions (page navigation)
  - Selection actions (dropdown menus)
  - Key press actions (keyboard shortcuts)
  - Navigation actions (goto URL)
  - Stop/None actions (task completion)

- **`help_functions.py`**: Helper utilities for action validation and parsing

## Usage

```python
from actions import create_click_action, create_type_action

# Create a click action
action = create_click_action(
    element_id="123",
    coords="<point>100 200</point>",
    description="search button",
    reasoning="Need to submit the search query"
)

# Create a type action
action = create_type_action(
    text="GUI Agent",
    element_id="456",
    coords="<point>50 100</point>",
    field_description="search input field",
    reasoning="Enter search term"
)
```

All actions follow the ActionTypes enumeration defined in `browser_env.actions`.

