import json
import re
from typing import Optional, Dict

def parse_action_json(message: str) -> Optional[Dict]:
    """
    Parses the action JSON from a ChatCompletionMessage content string.

    Args:
        message (str): The content string from a ChatCompletionMessage.

    Returns:
        dict or None: Parsed JSON dictionary if found, else None.
    """
    # Pattern to extract content after 'Action: '
    pattern = r'Action:\s*(\{.*\})'

    match = re.search(pattern, message)
    if match:
        try:
            action_json = json.loads(match.group(1))
            result = {'function_call': action_json}
            return result
        except Exception as e:
            # print(f"Failed to parse JSON: {e}")
            return message
    # ```json
    # {"name": "click", "arguments": {"description": "27", "reasoning": "I need to select 'Sydney, New South Wales, Australia' as the destination to book a flight."}}
    # ```
    pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, message)
    if matches:
        json_str = matches[0]
        try:
            action_json = json.loads(json_str)
            result = {'function_call': action_json}
            return result
        except Exception as e:
            # print(f"Failed to parse JSON: {e}")
            pass
    # pattern = r'```json\s*(\{.*\})\s*```'
    # match = re.search(pattern, message, re.DOTALL)
    if "```json" in message:
        message = message.split("```json")[1].split("```")[0].strip().strip('\n').strip('\\n')
        try:
            action_json = json.loads(message)
            result = {'function_call': action_json}
            return result
        except Exception as e:
            # print(f"Failed to parse JSON: {e}")
            return message
    try:
        action_json = json.loads(message)
        if isinstance(action_json, dict) and "name" in action_json and "arguments" in action_json:
            return {'function_call': action_json}
    except Exception as e:
        # print(f"Failed to parse JSON: {e}")
        return message
    return message