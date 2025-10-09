import json
import os
import sys
sys.path.append('GUI-Agent')

from memory.experience_memory import Memory
from data_preparation.help_functions import *
from tqdm import tqdm
import random
import argparse

negative_phrases = ['Early stop', 'cannot', 'not found', 'not available', "can't"]
columns = {
    'messages': 'json',
    'response': 'str',
    'task_id': 'str',
    'task_description': 'str',
    'similar_trajectories': 'json',
    'recent_trajectory': 'json'
}

training_data_path = 'training_data'
CLICK_NUM = 0
TYPE_NUM = 0
SCROLL_NUM = 0
STOP_NUM = 0
ANALYZE_NUM = 0

valid_file_nums_per_dataset = {}

def process_single_file(file_path, dataset, domain, model, memory, all_samples, existing_memory=False):
    global CLICK_NUM, TYPE_NUM, SCROLL_NUM, STOP_NUM, ANALYZE_NUM

    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f'Error loading {file_path}: {e}')
            return 0
        question = data['task_description']
        task_id = data['conversation_id']
        conversation_data = data['rounds']
        total_rounds = data['total_rounds']
        max_rounds = 10
        if dataset != 'mind2web' and domain != 'conversation':
            if total_rounds < 3 or total_rounds >= max_rounds:
                return 0
        final_answer = conversation_data[-1]['response']
        if any(phrase in final_answer for phrase in negative_phrases):
            print('Invalid Answer')
            return 0
        
        # Use the new Memory class for multimodal retrieval
        # Get current image from the first round for multimodal matching
        current_image = None
        if len(conversation_data) > 0:
            current_image = get_base64_image_from_conversation(conversation_data[0])
        
        # Retrieve similar conversations using the new Memory class
        question = f"{dataset}_{domain}: {question}"
        selected_conversations = memory.retrieve_similar_conversations(
            question, 
            current_image=current_image,
            model=model,
            similar_num=5
        )
        
        similar_trajectories = []
        file_id_list = []
        for conversation in selected_conversations:
            with open(conversation, 'r') as f:
                similar_data = json.load(f)
                similar_conversation_data = similar_data['rounds']
                if len(similar_conversation_data) < 2:
                    continue
                if existing_memory:
                    file_id_list.append(conversation.split('/')[-1].split('.')[0])
                    if len(file_id_list) == 3:
                        break
                else:
                    similar_actions_list, similar_images_list = organize_similar_tajectory(similar_conversation_data)
                    similar_trajectories.append({
                        'actions': similar_actions_list,
                        'images': similar_images_list
                    })
                    if len(similar_trajectories) == 3:
                        break
        print('file_id_list in process_single_file', file_id_list)
        
        history_trajectories = []
        for idx, single_round in enumerate(conversation_data):
            if 'click' in single_round['response']:
                CLICK_NUM += 1
            elif 'type' in single_round['response']:
                TYPE_NUM += 1
            elif 'scroll' in single_round['response']:
                SCROLL_NUM += 1
            elif 'stop' in single_round['response']:
                STOP_NUM += 1
            elif 'analyze' in single_round['response']:
                ANALYZE_NUM += 1
            if len(history_trajectories) < 5:
                recent_trajectory = history_trajectories
            else:
                recent_trajectory = history_trajectories[-5:]
            recent_trajectory = [item['actions'][0] for item in recent_trajectory]
                
            ####################
            with open('GUI-Agent/agent/prompts/system_prompt_simple.txt', 'r') as f:
                system_prompt = f.read()
            user_prompt = f"Your task is: {data['task_description']}. This is the current screenshot. Please give the next action in structured JSON format."
            if len(recent_trajectory) > 0:
                previous_actions = ""
                for history_id, actions in enumerate(recent_trajectory):
                    previous_actions += f"Round {history_id}: {actions}\n"
                history_prompt = f"Here are the recent actions you have taken: {previous_actions}. Please give the next action in structured JSON format."
            else:
                history_prompt = ""
            current_image = get_base64_image_from_conversation(single_round)
            assert (isinstance(current_image, str) and current_image.startswith('data:image/png;base64,')), f"Wrong Image Format {current_image[:100]}"
            new_messages = [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_prompt},
                        {'type': 'image_url', 'image_url': {'url': current_image}}
                    ]
                }
            ]
            if history_prompt != "":
                new_messages.append({
                    'role': 'user',
                    'content': history_prompt
                })
            if existing_memory:
                sample = {
                    'messages': new_messages,
                    'response': single_round['response'],
                    'task_id': f'{task_id}_{idx}',
                    'task_description': question,
                    'file_id_list': file_id_list,
                }
                all_samples.append(sample)
            elif similar_trajectories != []:
                sample = {
                'messages': new_messages,
                'response': single_round['response'],
                'task_id': f'{task_id}_{idx}',
                'task_description': question,
                'similar_trajectories': similar_trajectories,
                'recent_trajectory': []
                }
                all_samples.append(sample)
            
            image = get_base64_image_from_conversation(single_round)
            assert isinstance(image, str), f"Wrong Image Format {image}"
            action = single_round['response']
            if image and isinstance(action, str):
                history_trajectories.append({
                    'actions': [action],
                    'images': [image]
                })
        return 1
    
    
def load_trajectories_onfly(trajectory_path, max_samples=None, filter_by_dataset=None, existing_memory=False):
    # Parse command line arguments for memory configuration
    multimodal = True
    faiss_index_path = None
    
    # Initialize the Memory class with multimodal capabilities
    print(f"Initializing Memory class with multimodal={multimodal}")
    memory = Memory(
        training_data_path=trajectory_path,
        faiss_index_path=faiss_index_path,
        multimodal=multimodal
    )
    
    # Print available datasets and domains
    available_datasets = memory.get_available_datasets_and_domains()
    print("Available datasets and domains:")
    for dataset, domains in available_datasets.items():
        print(f"{dataset}: {domains}")
        valid_file_nums_per_dataset[dataset] = 0
    
    all_datasets = os.listdir(trajectory_path)
    all_datasets = [dataset for dataset in all_datasets if os.path.isdir(os.path.join(trajectory_path, dataset))]
    all_samples = []
    valid_files = 0

    for dataset in tqdm(all_datasets):
        if filter_by_dataset is not None and dataset not in filter_by_dataset:
            continue
        all_domains = os.listdir(f'{trajectory_path}/{dataset}')
        all_domains = [domain for domain in all_domains if os.path.isdir(f'{trajectory_path}/{dataset}/{domain}')]

        for domain in tqdm(all_domains):
            try:
                all_tests = os.listdir(f'{trajectory_path}/{dataset}/{domain}/qwen2.5-vl-32b')
                all_tests = [test for test in all_tests if os.path.isdir(f'{trajectory_path}/{dataset}/{domain}/qwen2.5-vl-32b/{test}')]
            except Exception as e:
                print(f'Error listing tests for {dataset} {domain}: {e}')
                continue
            seen_configs = set()

            for test in tqdm(all_tests):
                if 'test' not in test:
                    continue
                if not os.path.exists(f'{trajectory_path}/{dataset}/{domain}/qwen2.5-vl-32b/{test}/success'):
                    continue
                success_files = os.listdir(f'{trajectory_path}/{dataset}/{domain}/qwen2.5-vl-32b/{test}/success')
                all_files = [f'success/{file}' for file in success_files]  # + [f'positive/{file}' for file in positive_files]
                all_files = [file for file in all_files if file.endswith('.jsonl')]
                random.shuffle(all_files)
                print('*'*50, f'{dataset} {domain} {test}', '*'*50)
                for file in tqdm(all_files):
                    if file in seen_configs:
                        continue
                    file_path = f'{trajectory_path}/{dataset}/{domain}/qwen2.5-vl-32b/{test}/{file}'
                    valid_file_num = process_single_file(file_path, dataset, domain, 'qwen2.5-vl-32b', memory, all_samples, existing_memory)
                    valid_files += valid_file_num
                    valid_file_nums_per_dataset[dataset] += valid_file_num
                    seen_configs.add(file)
                    if max_samples is not None and valid_files >= max_samples:
                        print(f'Valid files: {valid_files}')
                        print(f'valid_file_nums_per_dataset: {valid_file_nums_per_dataset}')
                        return all_samples
                    
    with open('training_data_detail.txt', 'w') as f:
        f.write(f'Valid files: {valid_files}\n')
        f.write(f'CLICK_NUM: {CLICK_NUM}/{len(all_samples)}, {CLICK_NUM/len(all_samples)*100}%\n')
        f.write(f'TYPE_NUM: {TYPE_NUM}/{len(all_samples)}, {TYPE_NUM/len(all_samples)*100}%\n')
        f.write(f'SCROLL_NUM: {SCROLL_NUM}/{len(all_samples)}, {SCROLL_NUM/len(all_samples)*100}%\n')
        f.write(f'STOP_NUM: {STOP_NUM}/{len(all_samples)}, {STOP_NUM/len(all_samples)*100}%\n')
        f.write(f'ANALYZE_NUM: {ANALYZE_NUM}/{len(all_samples)}, {ANALYZE_NUM/len(all_samples)*100}%\n')
        f.write(f'valid_file_nums_per_dataset: {valid_file_nums_per_dataset}\n')
    return all_samples


if __name__ == '__main__':
    load_trajectories_onfly(trajectory_path='training_data', max_samples=10, filter_by_dataset=['expand_memory'])
    