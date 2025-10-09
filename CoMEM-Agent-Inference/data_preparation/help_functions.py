def get_content_safe(response):
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        response = response[0]
    if 'content' in response:
        return response['content']
    else:
        return response

def clean_round_history(round_list, memory):
    cleaned_round_list = []
    previous_action_name, previous_action_reasoning = None, None
    for round in round_list:
        action_json, current_action_name, current_action_reasoning = memory.parse_action_from_response(round['response'])
        if action_json:
            if current_action_name == previous_action_name and current_action_reasoning == previous_action_reasoning:
                continue
            else:
                round['action_json'] = str(action_json)
                round['action_name'] = current_action_name
                round['action_reasoning'] = current_action_reasoning
                cleaned_round_list.append(round)
                previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
        else:
            continue
        
    return cleaned_round_list

def get_base64_image_from_conversation(messages):
    if 'messages' in messages:
        messages = messages['messages']
    for msg in messages:
        try:
            if isinstance(msg.get('content'), list):
                for item in msg['content']:
                    if item.get('type') == 'image_url':
                        return item['image_url']['url']
        except:
            print(msg)
    return None

def organize_similar_tajectory(conversation_data):
    actions_list = []
    images_list = []

    for round in conversation_data:
        image = get_base64_image_from_conversation(round)
        assert isinstance(image, str), f"Wrong Image Format {image}"
        if image:
            images_list.append(image)
            actions_list.append(round['response'])
    if len(actions_list) >= 10:
        return actions_list[::2], images_list[::2]
    return actions_list, images_list