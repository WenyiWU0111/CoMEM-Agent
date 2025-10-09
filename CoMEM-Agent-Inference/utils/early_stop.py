"""Early stop functionality for the GUI Agent"""
from beartype import beartype
from browser_env import Trajectory, ActionTypes, Action


@beartype
def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    last_k_actions: list[Action]
    action_seq: list[Action]
    action_seq = [item for item in trajectory if item.get('action_type', '')!='']
    
    # reach the max step
    num_steps = len(action_seq)
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = action_seq[-k:]  # type: ignore[assignment]
    for idx, action in enumerate(last_k_actions):
        if action.get('action_type', '') == '':
            print(f"Action {idx} in last_k_actions is empty: {action}")
    if len(last_k_actions) >= k:
        if all(
            [
                action.get('action_type', '') == ActionTypes.NONE
                for action in last_k_actions
            ]
        ):
            return True, f"Failed to parse actions for {k} times"

    return False, "" 