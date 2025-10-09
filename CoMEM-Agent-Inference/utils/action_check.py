"""Action self-check and retry functionality for the GUI Agent"""
import logging
from typing import Callable
from browser_env.actions import ActionTypes

def action_self_check(gen_action: Callable, intent: str, page, trajectory, meta_data, max_retries: int = 3, repeat_threshold: int = 3):
    """
    Perform action self-check and retry if needed
    
    Args:
        gen_action: Function to generate action
        intent: Current intent
        page: Browser page object
        trajectory: Current trajectory
        max_retries: Maximum number of retries
        repeat_threshold: Threshold for repeating actions
        
    Returns:
        Generated action
    """
    logger = logging.getLogger("logger")
    trajectory_for_check = trajectory.copy()
    error_message = None
    
    for attempt in range(max_retries):
        # try:
        action, meta_data = gen_action(intent, meta_data, error_message)
    
        # Check if action is valid
        if action is None or (isinstance(action, dict) and action.get('action_type') == ''):
            if attempt < max_retries - 1:
                error_message = f"Failed to generate valid action. This is your *wrong response*: {meta_data['response_history'][-1]}. Please try again."
                continue
            else:
                logger.error("Failed to generate valid action after all retries")
                return action, meta_data
        
        # Check for repeating actions
        elif _is_repeating_action(trajectory_for_check, action, repeat_threshold):
            # logger.warning(f"Detected repeating action on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                # Add error feedback to encourage different action
                error_message = f"This action {meta_data['action_history'][-1]} was **repeated multiple times**. Please carefully check the current page and try a different approach!"
                trajectory_for_check.append(action)
                continue
            else:
                logger.error("Failed to generate non-repeating action after all retries")
                return action, meta_data
        
        else:
            break
        
    return action, meta_data
        


def _is_repeating_action(trajectory, action, threshold: int) -> bool:
    """
    Check if the action is being repeated too many times
    
    Args:
        trajectory: Current trajectory
        action: Current action
        threshold: Threshold for considering action as repeating
        
    Returns:
        True if action is repeating too much
    """
    # Extract actions from trajectory
    actions = [item for item in trajectory if isinstance(item, dict) and item.get('action_type', '') != '']
    
    if len(actions) < threshold:
        return False
    
    # Check if the last 'threshold' actions are the same
    recent_actions = actions[-threshold:]
    
    # Compare current action with recent actions
    for recent_action in recent_actions:
        if not _actions_equivalent(action, recent_action):
            return False
    
    return True


def _actions_equivalent(action1, action2) -> bool:
    """
    Check if two actions are equivalent
    
    Args:
        action1: First action
        action2: Second action
        
    Returns:
        True if actions are equivalent
    """
    if not isinstance(action1, dict) or not isinstance(action2, dict):
        return False
    
    # Compare action types
    if action1.get('action_type') != action2.get('action_type'):
        return False
    
    # For specific action types, compare additional properties
    action_type = action1.get('action_type')
    
    if action_type in [ActionTypes.CLICK, ActionTypes.TYPE]:
        # Compare coordinates for click actions
        if 'description' in action1 and 'description' in action2:
            return calculate_text_similarity(action1['description'], action2['description']) > 0.5
        # Compare text for type actions
        elif 'text' in action1 and 'text' in action2:
            return calculate_text_similarity(action1['text'], action2['text']) > 0.5
    
    return True 

def calculate_text_similarity(text1, text2):
    """
    Calculate the similarity between two text strings
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Similarity score between 0 and 1
    """
    words1 = set(text1.split())
    words2 = set(text2.split())
    if not words1 or not words2:
        return 1.0 if text1 == text2 else 0.0
    else:
        return len(words1.intersection(words2)) / max(len(words1), len(words2))
    