"""Action creator functions for the GUI Agent"""
from typing import Dict, Any, Optional
from browser_env import Action, create_stop_action as browser_create_stop_action, create_none_action as browser_create_none_action
from browser_env.actions import ActionTypes
from browser_env.constants import SPECIAL_KEY_MAPPINGS


def create_click_action(element_id: str, coords: str, description: str, reasoning: str = "") -> Action:
    """
    Create a click action based on description
    
    Args:
        element_id: Element id of the element to click
        coords: Coordinates of the element to click, in the format of "<point>x1 y1</point>", and it should be valid with two numbers, without any other text!
        description: Description of the element to click (e.g., "search button", "login link")
        reasoning: Reasoning for why this element should be clicked
        
    Returns:
        Action dictionary for clicking
    """
    return {
        'action_type': ActionTypes.CLICK,
        'element_id': element_id,
        'coords': coords,
        'description': description,
        'reasoning': reasoning
    }


def create_type_action(text: str, element_id: str, coords: str, field_description: str, reasoning: str = "") -> Action:
    """
    Create a type action based on field description
    
    Args:
        text: Text to type into the input field
        element_id: Element id of the input field
        coords: Coordinates of the element to type into, in the format of "<point>x1 y1</point>", and it should be valid with two numbers, without any other text!
        field_description: Description of the input field (e.g., "search box", "username field")
        reasoning: Reasoning for why this text should be typed in this field
        
    Returns:
        Action dictionary for typing
    """
    return {
        'action_type': ActionTypes.TYPE,
        'text': text,
        'element_id': element_id,
        'coords': coords,
        'field_description': field_description,
        'reasoning': reasoning
    }

def create_selection_action(element_id: str, coords: str, description: str, reasoning: str = "") -> Action:
    """
    Create a select action based on description
    """
    return {
        'action_type': ActionTypes.SELECT,
        'element_id': element_id,
        'coords': coords,
        'description': description,
        'reasoning': reasoning
    }

def create_scroll_action(direction: str, reasoning: str = "") -> Action:
    """
    Create a scroll action
    
    Args:
        direction: Direction to scroll (up, down, left, right)
        reasoning: Reasoning for why scrolling in this direction is needed
        
    Returns:
        Action dictionary for scrolling
    """
    return {
        'action_type': ActionTypes.SCROLL,
        'direction': direction,
        'reasoning': reasoning
    }


def create_wait_action(seconds: float = 2.0, reasoning: str = "") -> Action:
    """
    Create a wait action with default 2 seconds
    
    Args:
        seconds: Number of seconds to wait (default: 2.0)
        reasoning: Reasoning for why waiting is necessary
        
    Returns:
        Action dictionary for waiting
    """
    return {
        'action_type': ActionTypes.WAIT,
        'seconds': seconds,
        'reasoning': reasoning
    }


def create_stop_action(answer: str, reasoning: str = "") -> Action:
    """
    Create a stop action with answer
    
    Args:
        answer: Final answer or result of the task
        reasoning: Reasoning for why the task is complete
        
    Returns:
        Action dictionary for stopping
    """
    return browser_create_stop_action(answer)


def create_key_press_action(key_comb: str, reasoning: str = "") -> Action:
    """
    Create a key press action
    
    Args:
        key_comb: Combination of keys to press (e.g., "enter", "delete", "space")
        reasoning: Reasoning for why this key combination should be pressed
        
    Returns:
        Action dictionary for key press
    """
    mapping = SPECIAL_KEY_MAPPINGS.get(key_comb, 'Enter')
    
    return {
        'action_type': ActionTypes.KEY_PRESS,
        'key_comb': mapping,
        'reasoning': reasoning
    }


def create_goto_url_action(url: str) -> Action:
    """Create an action instructing the environment to navigate to a URL."""
    return {
        'action_type': ActionTypes.GOTO_URL,
        'url': url,
        'reasoning': f'Navigate to {url}'
    }


def create_none_action() -> Action:
    """
    Create a none action (no action to take)
    
    Returns:
        Action dictionary for no action
    """
    return browser_create_none_action()


def create_action_from_function_call(func_name: str, func_args: Dict[str, Any]) -> Action:
    """
    Create an action from function call parameters
    
    Args:
        func_name: Name of the function called
        func_args: Arguments passed to the function
        
    Returns:
        Action dictionary
    """
    if func_name == 'click':
        return create_click_action(
            element_id=func_args.get('element_id', ''),
            description=func_args.get('description', ''),
            reasoning=func_args.get('reasoning', '')
        )
    elif func_name == 'type':
        return create_type_action(
            text=func_args.get('text', ''),
            element_id=func_args.get('element_id', ''),
            field_description=func_args.get('field_description', ''),
            reasoning=func_args.get('reasoning', '')
        )
    elif func_name == 'scroll':
        return create_scroll_action(
            direction=func_args.get('direction', 'down'),
            reasoning=func_args.get('reasoning', '')
        )
    elif func_name == 'wait':
        return create_wait_action(
            seconds=2.0,  # Default as per requirements
            reasoning=func_args.get('reasoning', '')
        )
    elif func_name == 'stop':
        return create_stop_action(
            answer=func_args.get('answer', 'Task completed'),
            reasoning=func_args.get('reasoning', '')
        )
    elif func_name == 'select':
        return create_select_action(
            element_id=func_args.get('element_id', ''),
            description=func_args.get('description', ''),
            text=func_args.get('text', ''),
            reasoning=func_args.get('reasoning', '')
        )
    else:
        return create_none_action()


def validate_action(action: Action) -> bool:
    """
    Validate if an action is appropriate
    
    Args:
        action: Action to validate
        
    Returns:
        True if action is valid, False otherwise
    """
    if not action or action.get('action_type') == '':
        return False
    
    action_type = action.get('action_type')
    
    if action_type == ActionTypes.CLICK:
        # Check if description is provided
        description = action.get('description', '')
        return bool(description.strip())
    
    elif action_type == ActionTypes.TYPE:
        # Check if text and field description are provided
        text = action.get('text', '')
        field_description = action.get('field_description', '')
        return bool(text.strip() and field_description.strip())
    
    elif action_type == ActionTypes.SCROLL:
        # Check if direction is valid
        valid_directions = ['up', 'down', 'left', 'right']
        direction = action.get('direction', '')
        return direction in valid_directions
    
    elif action_type == 'wait':
        # Wait actions are always valid
        return True
    
    elif action_type == ActionTypes.STOP:
        # Stop actions are always valid
        return True
    
    elif action_type == ActionTypes.SELECT:
        # Select actions are always valid
        return True
    
    else:
        return False 