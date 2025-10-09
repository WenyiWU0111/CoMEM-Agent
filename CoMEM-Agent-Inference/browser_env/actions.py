"""
Simplified browser environment action space for GUI Agent.
"""
import random
import string
from enum import IntEnum
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from beartype import beartype
from playwright._impl._api_structures import ViewportSize
from playwright.sync_api import BrowserContext, Locator, Page

from browser_env.constants import (
    ASCII_CHARSET,
    MAX_ELEMENT_INDEX_IN_VIEWPORT,
    MAX_PAGE_NUMBER,
    ROLES,
    SPECIAL_KEY_MAPPINGS,
    RolesType,
)
from browser_env.processors import ObservationProcessor


@beartype
def is_in_viewport(
    element: Locator, viewport: ViewportSize, threshold: float = 0.3
) -> bool:
    """Given a playwright locator, check if it is in the viewport"""
    box = element.bounding_box()
    assert box is not None
    boxx0 = box["x"]
    boxx1 = box["x"] + box["width"]
    boxy0 = box["y"]
    boxy1 = box["y"] + box["height"]
    viewportx0, viewporty0 = 0, 0
    viewportx1, viewporty1 = viewport["width"], viewport["height"]
    inter = max(0, min(boxx1, viewportx1) - max(boxx0, viewportx0)) * max(
        0, min(boxy1, viewporty1) - max(boxy0, viewporty0)
    )
    ratio = inter / (box["width"] * box["height"])
    return ratio > threshold


class Action(TypedDict):
    action_type: int
    coords: npt.NDArray[np.float32]
    element_role: int
    element_name: str
    text: list[int]
    page_number: int
    url: str
    nth: int
    element_id: str
    direction: str
    key_comb: str
    pw_code: str
    answer: str
    raw_prediction: str  # raw prediction from the model


@beartype
def action2str(
    action: Action, action_set_tag: str, semantic_element: str = ""
) -> str:
    """Convert action to string representation."""
    action_type = action["action_type"]
    match action_type:
        case ActionTypes.NONE:
            return "none"
        case ActionTypes.SCROLL:
            direction = action["direction"]
            return f"scroll {direction}"
        case ActionTypes.KEY_PRESS:
            key_comb = action["key_comb"]
            return f"key_press {key_comb}"
        case ActionTypes.MOUSE_CLICK:
            coords = action["coords"]
            return f"mouse_click {coords[0]:.2f} {coords[1]:.2f}"
        case ActionTypes.KEYBOARD_TYPE:
            text = action["text"]
            return f"keyboard_type {text}"
        case ActionTypes.CLICK:
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"click {element_role} {element_name}"
        case ActionTypes.TYPE:
            text = action["text"]
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"type {text} {element_role} {element_name}"
        case ActionTypes.HOVER:
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"hover {element_role} {element_name}"
        case ActionTypes.PAGE_FOCUS:
            page_number = action["page_number"]
            return f"page_focus {page_number}"
        case ActionTypes.NEW_TAB:
            return "new_tab"
        case ActionTypes.GO_BACK:
            return "go_back"
        case ActionTypes.GO_FORWARD:
            return "go_forward"
        case ActionTypes.GOTO_URL:
            url = action["url"]
            return f"goto_url {url}"
        case ActionTypes.PAGE_CLOSE:
            return "page_close"
        case ActionTypes.STOP:
            answer = action["answer"]
            return f"stop {answer}"
        case _:
            return "unknown"


def action2create_function(action: Action) -> str:
    """Convert action to create function call string."""
    action_type = action["action_type"]
    match action_type:
        case ActionTypes.NONE:
            return "create_none_action()"
        case ActionTypes.SCROLL:
            direction = action["direction"]
            return f"create_scroll_action('{direction}')"
        case ActionTypes.KEY_PRESS:
            key_comb = action["key_comb"]
            return f"create_key_press_action('{key_comb}')"
        case ActionTypes.MOUSE_CLICK:
            coords = action["coords"]
            return f"create_mouse_click_action({coords[0]:.2f}, {coords[1]:.2f})"
        case ActionTypes.KEYBOARD_TYPE:
            text = action["text"]
            return f"create_keyboard_type_action({text})"
        case ActionTypes.CLICK:
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"create_click_action(element_role='{element_role}', element_name='{element_name}')"
        case ActionTypes.TYPE:
            text = action["text"]
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"create_type_action('{text}', element_role='{element_role}', element_name='{element_name}')"
        case ActionTypes.HOVER:
            element_role = action["element_role"]
            element_name = action["element_name"]
            return f"create_hover_action(element_role='{element_role}', element_name='{element_name}')"
        case ActionTypes.PAGE_FOCUS:
            page_number = action["page_number"]
            return f"create_page_focus_action({page_number})"
        case ActionTypes.NEW_TAB:
            return "create_new_tab_action()"
        case ActionTypes.GO_BACK:
            return "create_go_back_action()"
        case ActionTypes.GO_FORWARD:
            return "create_go_forward_action()"
        case ActionTypes.GOTO_URL:
            url = action["url"]
            return f"create_goto_url_action('{url}')"
        case ActionTypes.PAGE_CLOSE:
            return "create_page_close_action()"
        case ActionTypes.STOP:
            answer = action["answer"]
            return f"create_stop_action('{answer}')"
        case _:
            return "create_none_action()"


class ActionTypes(IntEnum):
    """Valid action types for browser env."""

    NONE = 0
    SCROLL = 1
    KEY_PRESS = 2
    MOUSE_CLICK = 3
    KEYBOARD_TYPE = 4
    MOUSE_HOVER = 5
    CLICK = 6
    TYPE = 7
    HOVER = 8
    PAGE_FOCUS = 9
    NEW_TAB = 10
    GO_BACK = 11
    GO_FORWARD = 12
    GOTO_URL = 13
    PAGE_CLOSE = 14
    WAIT = 15
    STOP = 17
    SELECT = 18
    def __str__(self) -> str:
        return f"ACTION_TYPES.{self.name}"


def is_equivalent(a: Action, b: Action) -> bool:
    """Return True if two actions are equal."""
    a_type = a.get("action_type", None)
    b_type = b.get("action_type", None)
    
    if isinstance(a_type, str) and isinstance(b_type, str):
        return a_type == b_type
    
    if a_type != b_type:
        return False
        
    match (a_type):
        case ActionTypes.NONE:
            return True
        case ActionTypes.SCROLL:
            da = "up" if "up" in a["direction"] else "down"
            db = "up" if "up" in b["direction"] else "down"
            return da == db
        case ActionTypes.KEY_PRESS:
            return a["key_comb"] == b["key_comb"]
        case ActionTypes.MOUSE_CLICK | ActionTypes.MOUSE_HOVER:
            return np.allclose(a["coords"], b["coords"])
        case ActionTypes.KEYBOARD_TYPE:
            return a["text"] == b["text"]
        case ActionTypes.CLICK | ActionTypes.HOVER | ActionTypes.TYPE:
            if a_type and b_type:
                return a_type == b_type
            else:
                return (a["element_role"] == b["element_role"] and 
                       a["element_name"] == b["element_name"] and 
                       a["nth"] == b["nth"])
        case ActionTypes.PAGE_FOCUS:
            return a["page_number"] == b["page_number"]
        case ActionTypes.GOTO_URL:
            return a["url"] == b["url"]
        case ActionTypes.STOP:
            return a["answer"] == b["answer"]
        case _:
            return False

def create_random_action() -> Action:
    """Create a random action for testing."""
    action_type = random.choice(list(ActionTypes))
    action = {
        "action_type": action_type,
        "coords": np.array([random.random(), random.random()], dtype=np.float32),
        "element_role": random.randint(0, len(ROLES) - 1),
        "element_name": "".join(random.choices(string.ascii_letters, k=10)),
        "text": [random.randint(0, len(ASCII_CHARSET) - 1) for _ in range(5)],
        "page_number": random.randint(0, MAX_PAGE_NUMBER - 1),
        "url": "https://example.com",
        "nth": random.randint(0, MAX_ELEMENT_INDEX_IN_VIEWPORT - 1),
        "element_id": "".join(random.choices(string.ascii_letters, k=10)),
        "direction": random.choice(["up", "down"]),
        "key_comb": "Enter",
        "pw_code": "",
        "answer": "random answer",
        "raw_prediction": "random prediction",
    }
    return action


@beartype
def create_none_action() -> Action:
    """Create a none action."""
    return {
        "action_type": ActionTypes.NONE,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_stop_action(answer: str) -> Action:
    """Create a stop action with answer."""
    return {
        "action_type": ActionTypes.STOP,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": answer,
        "raw_prediction": "",
    }


@beartype
def create_scroll_action(direction: str) -> Action:
    """Create a scroll action."""
    return {
        "action_type": ActionTypes.SCROLL,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": direction,
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_wait_action(seconds: float = 2.0) -> Action:
    """Create a wait action."""
    return {
        "action_type": ActionTypes.WAIT,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_mouse_hover_action(
    left: float | None = None, top: float | None = None
) -> Action:
    """Create a mouse hover action."""
    if left is None:
        left = random.random()
    if top is None:
        top = random.random()
    return {
        "action_type": ActionTypes.MOUSE_HOVER,
        "coords": np.array([left, top], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_key_press_action(key_comb: str) -> Action:
    """Create a key press action."""
    def map_keys(key_comb: str) -> str:
        """Map special keys to their representations."""
        if key_comb in SPECIAL_KEY_MAPPINGS:
            return SPECIAL_KEY_MAPPINGS[key_comb]
        return key_comb

    return {
        "action_type": ActionTypes.KEY_PRESS,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": map_keys(key_comb),
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_page_focus_action(page_number: int) -> Action:
    """Create a page focus action."""
    return {
        "action_type": ActionTypes.PAGE_FOCUS,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": page_number,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_new_tab_action() -> Action:
    """Create a new tab action."""
    return {
        "action_type": ActionTypes.NEW_TAB,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_go_back_action() -> Action:
    """Create a go back action."""
    return {
        "action_type": ActionTypes.GO_BACK,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_go_forward_action() -> Action:
    """Create a go forward action."""
    return {
        "action_type": ActionTypes.GO_FORWARD,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_goto_url_action(url: str) -> Action:
    """Create a goto URL action."""
    return {
        "action_type": ActionTypes.GOTO_URL,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": url,
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_page_close_action() -> Action:
    """Create a page close action."""
    return {
        "action_type": ActionTypes.PAGE_CLOSE,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_mouse_click_action(
    left: float | None = None, top: float | None = None
) -> Action:
    """Create a mouse click action."""
    if left is None:
        left = random.random()
    if top is None:
        top = random.random()
    return {
        "action_type": ActionTypes.MOUSE_CLICK,
        "coords": np.array([left, top], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_keyboard_type_action(keys: list[int | str] | str) -> Action:
    """Create a keyboard type action."""
    if isinstance(keys, str):
        text = [ord(c) for c in keys]
    else:
        text = [int(k) if isinstance(k, str) else k for k in keys]
    return {
        "action_type": ActionTypes.KEYBOARD_TYPE,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": text,
        "page_number": 0,
        "url": "",
        "nth": 0,
        "element_id": "",
        "direction": "",
        "key_comb": "",
        "pw_code": "",
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_click_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
    coords: str = "",
) -> Action:
    """Create a click action."""
    return {
        "action_type": ActionTypes.CLICK,
        "coords": coords,
        "element_role": ROLES.index(element_role),
        "element_name": element_name,
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": nth,
        "element_id": element_id,
        "direction": "",
        "key_comb": "",
        "pw_code": pw_code,
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_hover_action(
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
) -> Action:
    """Create a hover action."""
    return {
        "action_type": ActionTypes.HOVER,
        "coords": np.array([0.0, 0.0], dtype=np.float32),
        "element_role": ROLES.index(element_role),
        "element_name": element_name,
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": nth,
        "element_id": element_id,
        "direction": "",
        "key_comb": "",
        "pw_code": pw_code,
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def create_type_action(
    text: str,
    element_id: str = "",
    element_role: RolesType = "link",
    element_name: str = "",
    pw_code: str = "",
    nth: int = 0,
    coords: str = "",
) -> Action:
    """Create a type action."""
    return {
        "action_type": ActionTypes.TYPE,
        "coords": coords,
        "element_role": ROLES.index(element_role),
        "element_name": element_name,
        "text": [ord(c) for c in text],
        "page_number": 0,
        "url": "",
        "nth": nth,
        "element_id": element_id,
        "direction": "",
        "key_comb": "",
        "pw_code": pw_code,
        "answer": "",
        "raw_prediction": "",
    }


@beartype
def execute_scroll(direction: str, page: Page) -> None:
    """Execute scroll action."""
    if direction == "up":
        page.mouse.wheel(0, -500)
    elif direction == "down":
        page.mouse.wheel(0, 500)
    elif direction == "left":
        page.mouse.wheel(-500, 0)
    elif direction == "right":
        page.mouse.wheel(500, 0)


@beartype
def execute_key_press(key: str, page: Page) -> None:
    """Execute key press action."""
    page.keyboard.press(key)


@beartype
def execute_mouse_hover(left: float, top: float, page: Page) -> None:
    """Execute mouse hover action."""
    page.mouse.move(left, top)


def execute_mouse_click(left: float, top: float, page: Page) -> None:
    """Execute mouse click action."""
    page.mouse.click(left, top)


@beartype
def execute_keyboard_type(text: str, page: Page) -> None:
    """Execute keyboard type action."""
    page.keyboard.type(text)


@beartype
def execute_click_current(page: Page) -> None:
    """Execute click on current element."""
    page.click("body")


@beartype
def execute_type(keys: list[int], page: Page) -> None:
    """Execute type action with key codes."""
    text = "".join([chr(k) for k in keys])
    page.keyboard.type(text)


@beartype
def execute_action(
    action: Action,
    page: Page,
    browser_ctx: BrowserContext,
    observation_processor: ObservationProcessor,
) -> Page:
    """Execute an action on the page."""
    action_type = action["action_type"]
    
    match action_type:
        case ActionTypes.NONE:
            pass
        case ActionTypes.SCROLL:
            direction = action["direction"]
            execute_scroll(direction, page)
        case ActionTypes.KEY_PRESS:
            key_comb = action["key_comb"]
            execute_key_press(key_comb, page)
        case ActionTypes.MOUSE_CLICK:
            coords = action["coords"]
            execute_mouse_click(coords[0], coords[1], page)
        case ActionTypes.KEYBOARD_TYPE:
            text = "".join([chr(k) for k in action["text"]])
            execute_keyboard_type(text, page)
        case ActionTypes.MOUSE_HOVER:
            coords = action["coords"]
            execute_mouse_hover(coords[0], coords[1], page)
        case ActionTypes.CLICK:
            execute_click_current(page)
        case ActionTypes.TYPE:
            text = "".join([chr(k) for k in action["text"]])
            execute_keyboard_type(text, page)
        case ActionTypes.HOVER:
            execute_click_current(page)  # Simplified hover
        case ActionTypes.PAGE_FOCUS:
            page_number = action["page_number"]
            pages = list(browser_ctx.pages)
            if page_number < len(pages):
                pages[page_number].bring_to_front()
        case ActionTypes.NEW_TAB:
            browser_ctx.new_page()
        case ActionTypes.GO_BACK:
            page.go_back()
        case ActionTypes.GO_FORWARD:
            page.go_forward()
        case ActionTypes.GOTO_URL:
            url = action["url"]
            page.goto(url)
        case ActionTypes.PAGE_CLOSE:
            page.close()
        case ActionTypes.STOP:
            pass  # Stop action doesn't execute anything
        case _:
            pass
    
    return page


@beartype
class ActionParsingError(Exception):
    """Exception for action parsing errors."""
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)
    