import json
import os
import sys
sys.path.append('GUI-Agent')

from memory.experience_memory import Memory
# from streaming import MDSWriter
from data_preparation.help_functions import *
from tqdm import tqdm
import random
import base64
from PIL import Image
from io import BytesIO
import torch

negative_phrases = ['Early stop', 'cannot', 'not found', 'not available', "can't"]
columns = {
    'messages': 'json',
    'response': 'str',
    'task_id': 'str',
    'task_description': 'str',
    'similar_trajectories': 'json',
    'recent_trajectory': 'json'
}

similar_conversations_set = set()

path = 'CoMEM-Agent/CoMEM-Agent-train'
sys.path.append(path)
from src_agent.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
from transformers import AutoProcessor
compress_model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
    'WenyiWU0111/lora_qformer_test_V4-700_merged',
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

def knowledge_processor_vlm(processor, inputs, texts=None, images=None):
    """Process experience information for VLM"""
    
    # Default tokens for image processing
    DEFAULT_IM_START_TOKEN = "<|im_start|>"
    DEFAULT_IM_END_TOKEN = "<|im_end|>"
    DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
    VISION_START_TOKEN = "<|vision_start|>"
    VISION_END_TOKEN = "<|vision_end|>"
    
    all_experience_input_ids = [] 
    all_experience_pixel_values = []
    all_experience_image_grid_thw = []
    for trajectory_actions, trajectory_images in zip(texts, images):
        trajectory_text = ""
        trajectory_image = []
        for action, image_base64 in zip(trajectory_actions, trajectory_images):
            if isinstance(image_base64, dict) and image_base64.get('url', '').startswith('data:image/png;base64,'):
                image_bytes = base64.b64decode(image_base64.get('url', '').split(',')[1])
            elif isinstance(image_base64, str) and image_base64.startswith('data:image/png;base64,'):
                image_bytes = base64.b64decode(image_base64.split(',')[1])
            else:
                image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))
            trajectory_image.append(image)
            trajectory_text += f"{DEFAULT_IM_START_TOKEN}user\n{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}{action}{DEFAULT_IM_END_TOKEN}\n"
        if trajectory_image:
            e_inputs = processor(text=[trajectory_text], images=trajectory_image, padding=False, return_tensors='pt')
            e_input_ids = e_inputs['input_ids'].squeeze(0)
            e_pixel_values = e_inputs['pixel_values']
            e_image_grid_thw = e_inputs['image_grid_thw']
            all_experience_pixel_values.append(e_pixel_values)
            all_experience_image_grid_thw.append(e_image_grid_thw)
        else:
            e_input_ids = processor.tokenizer(trajectory_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
        
        all_experience_input_ids.append(e_input_ids)
        
    inputs['experience_input_ids'] = all_experience_input_ids
    inputs['experience_pixel_values'] = all_experience_pixel_values
    inputs['experience_image_grid_thw'] = all_experience_image_grid_thw
    
    return inputs

def process_vision_info(conversation):
    """Process vision information from conversation"""
    image_inputs = []
    
    for message in conversation:
        if isinstance(message['content'], list):
            for item in message['content']:
                if item['type'] == 'image_url':
                    image_url = item['image_url']['url']
                    image_bytes = base64.b64decode(image_url.split(',')[1])
                    image = Image.open(BytesIO(image_bytes))
                    image_inputs.append(image)
    
    return image_inputs
    
def get_knowledge_embeddings(processor, compress_model, image, prompt, experience_texts, experience_images, file_id_list):
    inputs = {}
    
    # Process experience information
    inputs_with_experience = knowledge_processor_vlm(
        processor=processor,
        inputs=inputs,
        texts=experience_texts,
        images=experience_images,
    )
    print('start to save memory embeddings')
    compress_model.save_memory_embeddings(input_ids=inputs_with_experience['experience_input_ids'], pixel_values=inputs_with_experience['experience_pixel_values'], image_grid_thws=inputs_with_experience['experience_image_grid_thw'], 
                                 file_id_list=file_id_list)

def process_single_file(question, dataset, domain, model, memory):
    
    # Retrieve similar conversations using the new Memory class
    question = f"{dataset}_{domain}: {question}"
    selected_conversations = memory.retrieve_similar_conversations(
        question, 
        current_image=None,
        model=model,
        similar_num=15
    )

    experience_texts = []
    experience_images = []
    file_id_list = []
    for conversation in selected_conversations:
        similar_conversations_set.add(conversation)
        with open(conversation, 'r') as f:
            similar_data = json.load(f)
            similar_conversation_data = similar_data['rounds']
            if len(similar_conversation_data) < 2:
                continue
            similar_actions_list, similar_images_list = organize_similar_tajectory(similar_conversation_data)
            experience_texts.append(similar_actions_list)
            experience_images.append(similar_images_list)
            file_id_list.append(conversation.split('/')[-1].split('.')[0])
            if len(experience_texts) == 10 and len(experience_images) == 10:
                break
    
    get_knowledge_embeddings(processor, compress_model, None, question, experience_texts, experience_images, file_id_list)
    
    print(f'Processed {question}')
    
def get_inference_memory_embeddings(trajectory_path, dataset, domain):
    # Parse command line arguments for memory configuration
    multimodal = False
    faiss_index_path = ''
    
    # Initialize the Memory class with multimodal capabilities
    print(f"Initializing Memory class with multimodal={multimodal}")
    memory = Memory(
        training_data_path=trajectory_path,
        faiss_index_path=faiss_index_path,
        multimodal=multimodal
    )
    
    if dataset == 'mmina':
        config_path_list = [path for path in os.listdir(f'mmina/{domain}') if path.endswith('.json')]
        for config_path in config_path_list:
            with open(f'mmina/{domain}/{config_path}', 'r') as f:
                config = json.load(f)
                question = config['intent']
                process_single_file(question, dataset, domain, 'qwen2.5-vl-32b', memory)
    elif dataset == 'mind2web':
        config_path_list = [path for path in os.listdir(f'Mind2Web/evaluation_data/{domain}') if path.endswith('.json')][:100]
        for config_path in config_path_list:
            with open(f'Mind2Web/evaluation_data/{domain}/{config_path}', 'r') as f:
                config = json.load(f)
                question = config['intent']
                process_single_file(question, dataset, domain, 'qwen2.5-vl-32b', memory)
    elif dataset == 'webvoyager':
        config_path_list = [path for path in os.listdir(f'webvoyager_evaluation/data/{domain}') if path.endswith('.json')]
        for config_path in config_path_list:
            with open(f'webvoyager_evaluation/data/{domain}/{config_path}', 'r') as f:
                config = json.load(f)
                question = config['intent']
                process_single_file(question, dataset, domain, 'qwen2.5-vl-32b', memory)

if __name__ == '__main__':
    get_inference_memory_embeddings(trajectory_path='training_data', dataset='mind2web', domain='test_domain_Service')
    
    