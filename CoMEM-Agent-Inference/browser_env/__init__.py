"""Simplified browser environment for GUI Agent"""

from .actions import (
    Action,
    ActionParsingError,
    ActionTypes,
    action2create_function,
    action2str,
    create_click_action,
    create_go_back_action,
    create_go_forward_action,
    create_goto_url_action,
    create_hover_action,
    create_key_press_action,
    create_keyboard_type_action,
    create_mouse_click_action,
    create_mouse_hover_action,
    create_new_tab_action,
    create_none_action,
    create_page_close_action,
    create_page_focus_action,
    create_random_action,
    create_scroll_action,
    create_stop_action,
    create_type_action,
    is_equivalent,
)
from .envs import ScriptBrowserEnv
from .processors import (
    SimpleImageObservationProcessor,
    SimpleTextObservationProcessor,
    ObservationProcessor,
    ObservationMetadata,
)
from .trajectory import Trajectory
from .utils import DetachedPage, StateInfo, Observation

__all__ = [
    # Environment
    "ScriptBrowserEnv",
    
    # Observation Processors
    "SimpleImageObservationProcessor",
    "SimpleTextObservationProcessor", 
    "ObservationProcessor",
    
    # Data Types
    "DetachedPage",
    "StateInfo",
    "Observation",
    "ObservationMetadata",
    "Action",
    "ActionTypes",
    "Trajectory",
    
    # Action Creation Functions
    "action2str",
    "create_random_action",
    "is_equivalent",
    "create_mouse_click_action",
    "create_mouse_hover_action",
    "create_none_action",
    "create_keyboard_type_action",
    "create_page_focus_action",
    "create_new_tab_action",
    "create_go_back_action",
    "create_go_forward_action",
    "create_goto_url_action",
    "create_page_close_action",
    "action2create_function",
    "create_scroll_action",
    "create_key_press_action",
    "create_click_action",
    "create_type_action",
    "create_hover_action",
    "create_stop_action",
    "ActionParsingError",
]
