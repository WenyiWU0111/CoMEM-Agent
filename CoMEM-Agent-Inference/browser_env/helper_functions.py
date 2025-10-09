import json
import re
from pathlib import Path
from typing import Any

from browser_env import (
    Action,
    StateInfo,
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def get_render_action(action: Action) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    action_str = ""
    from .action_parser_ground import get_action_description as ground_get_action_description
    if not isinstance(action, list):
        action = [action]
    for per_action in action:
        full_text = per_action.get('text', 'None') if isinstance(per_action.get('text', 'None'), str) else ''
        full_text += per_action.get('answer', 'None')
        action_str += f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{full_text}</pre></div>"
        # action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
        action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{ground_get_action_description(per_action)}</pre></div>"

    return action_str


def get_action_description(action: Action) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    from .action_parser_ground import get_action_description as ground_get_action_description
    if isinstance(action, list):
        action_str = '; '.join(ground_get_action_description(per_action) for per_action in action)
    else:
        action_str = ground_get_action_description(action)

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, config_file: str, result_dir: str
    ) -> None:
        with open(config_file, "r") as f:
            _config = json.load(f)
        
        _config_str = ""
        for k, v in _config.items():
            _config_str += f"{k}: {v}\n"
        _config_str = f"<pre>{_config_str}</pre>\n"
        task_id = _config["task_id"]

        self.render_file = open(
            Path(result_dir) / f"render_{task_id}.html", "a+"
        )
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<h2>New Page</h2>\n"
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>URL: {state_info['info']['page'].url}</a></h3>\n"
        new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_obs = observation["image"]
            image_str = img_obs
            new_content += f"<img src='data:image/png;base64,{image_str}' style='width:50vw; height:auto;'/>\n"
        # plan
        if meta_data and 'step_plan' in meta_data:
            step_plan = meta_data['step_plan']
            new_content += f"<div class='step_plan'><pre>{step_plan}</pre></div>\n"
        # response
        if meta_data and 'response_history' in meta_data:
            response = meta_data['response_history'][-1]
            new_content += f"<div class='response_history'><pre>{response}</pre></div>\n"
        # action
        action_str = get_render_action(action)
        # with yellow background
        action_str = f"<div class='predict_action'>{action_str}</div>"
        new_content += f"{action_str}\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        html_body = re.findall(r"<body>(.*?)</body>", html, re.DOTALL)[0]
        html_body += new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def close(self) -> None:
        self.render_file.close()
