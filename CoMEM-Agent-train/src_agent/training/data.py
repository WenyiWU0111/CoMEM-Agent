import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image
import re
import pickle
from PIL import Image

from params import DataArguments
from constants import *

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from streaming import MDSWriter, StreamingDataset
import shutil
import base64 
import uuid
import subprocess
from itertools import islice
import random

def load_mds(mds_dir):
    #subprocess.run(f"rm -f /dev/shm/*_locals", shell=True)
    dataset = StreamingDataset(local=mds_dir,
                           remote=None,
                           shuffle=False,
                           batch_size=1)
    records = []
    need_domain = ['mind2web_train', 'expand_memory']
    for sample in tqdm(dataset, desc="Loading MDS files"):
    # for sample in tqdm(islice(dataset, 100), desc="Loading MDS files", total=100):
        if any(domain in sample['task_description'] for domain in need_domain):
            records.append(sample)
    return records

def load_trajectories_direct(trajectory_path):
    """
    Load training data directly from trajectory directories.
    This is the new preferred method that avoids intermediate storage.
    """
    import sys
    sys.path.insert(0, 'CoMEM-Agent-Inference')
    from data_preparation.prepare_training_data_onfly import load_trajectories_onfly
    records = load_trajectories_onfly(
        trajectory_path,
        max_samples=None,
        filter_by_dataset=['mind2web_train', 'expand_memory']
    )
    random.shuffle(records)
    return records

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixel": min_pixel,
                "max_pixel": max_pixel

            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, min_pixels, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    return video_input[0], video_kwargs

list_data_dict = load_trajectories_direct(trajectory_path='training_data')

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        
        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.video_min_pixel = data_args.video_min_pixels
        self.video_max_pixel = data_args.video_max_pixels
        self.fps = data_args.fps

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        processor = self.processor

        # Extract data from the new structure
        messages = sources['messages']
        response = sources['response']
        similar_trajectories = sources.get('similar_trajectories', [])
        if len(similar_trajectories) < 3:
            existing_num = len(similar_trajectories)
            for _ in range(3 - existing_num):
                similar_trajectories.append(similar_trajectories[-1])
        # recent_trajectory = sources.get('recent_trajectory', [])
        recent_trajectory = []

        # Initialize lists for collecting data
        all_input_ids = []
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_second_gird = []

        # Process main conversation (messages + response)
        # Convert messages to the format expected by the processor
        conversation_text = ""
        conversation_images = []
        
        for message in messages:
            role = message['role']
            content = message['content']
            
            if isinstance(content, str):
                # Text-only content
                conversation_text += f"{DEFAULT_IM_START_TOKEN}{role}\n{content}{DEFAULT_IM_END_TOKEN}\n"
            elif isinstance(content, list):
                # Multi-modal content (text + images)
                message_text = f"{DEFAULT_IM_START_TOKEN}{role}\n"
                
                for item in content:
                    if item['type'] == 'text':
                        message_text += item['text']
                    elif item['type'] == 'image_url':
                        # Handle base64 image
                        image_url = item['image_url']['url']
                        if image_url.startswith('data:image/png;base64,'):
                            base64_data = image_url.split(',')[1]
                            image_bytes = base64.b64decode(base64_data)
                            image = Image.open(BytesIO(image_bytes))
                            conversation_images.append(image)
                            message_text += DEFAULT_IMAGE_TOKEN
                
                conversation_text += (message_text + f"{DEFAULT_IM_END_TOKEN}\n")
        
        # Add the response
        conversation_text += f"{DEFAULT_IM_START_TOKEN}assistant\n{response}{DEFAULT_IM_END_TOKEN}\n"

        # Process the main conversation
        if conversation_images:
            # Has images
            inputs = processor(text=[conversation_text], images=conversation_images, padding=False, return_tensors='pt')
            input_ids = inputs['input_ids'].squeeze(0)
            all_pixel_values.append(inputs['pixel_values'])
            all_image_grid_thw.append(inputs['image_grid_thw'])
        else:
            # Text only
            input_ids = processor.tokenizer(conversation_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)

        # Create labels (ignore user parts, only train on assistant response)
        # Find where the assistant response starts
        assistant_start = conversation_text.rfind(f"{DEFAULT_IM_START_TOKEN}assistant\n")
        if assistant_start != -1:
            assistant_text = conversation_text[assistant_start:]
            response_input_ids = processor.tokenizer(assistant_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            print('assistant_text', assistant_text)
            # Calculate the length of the prompt part (everything before assistant response)
            prompt_length = len(input_ids) - len(response_input_ids)
            
            # Create labels: ignore prompt part, only train on assistant response
            labels = torch.cat([
                torch.tensor([IGNORE_INDEX] * prompt_length),
                response_input_ids,
            ], dim=0)
        else:
            # If no assistant response found, ignore everything
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        all_input_ids.append(input_ids)
        all_labels.append(labels)

        # Process recent trajectory as history
        all_history_input_ids = []
        all_history_pixel_values = []
        all_history_image_grid_thw = []
        
        history_text = ""
        history_images = []
        for step in recent_trajectory[:3]:  # Limit to 3 recent steps
            action = step['actions']
            image_base64 = step['images']
            
            # Decode base64 image
            if isinstance(image_base64, dict) and image_base64.get('url', '').startswith('data:image/png;base64,'):
                image_bytes = base64.b64decode(image_base64.get('url', '').split(',')[1])
            elif isinstance(image_base64, str) and image_base64.startswith('data:image/png;base64,'):
                image_bytes = base64.b64decode(image_base64.split(',')[1])
            else:
                image_bytes = base64.b64decode(image_base64)
                
            image = Image.open(BytesIO(image_bytes))
            history_images.append(image)
            history_text += f"{DEFAULT_IM_START_TOKEN}user\n{DEFAULT_IMAGE_TOKEN}{action}{DEFAULT_IM_END_TOKEN}\n"
            
        h_inputs = processor(text=[history_text], images=history_images, padding=False, return_tensors='pt')
        h_input_ids = h_inputs['input_ids'].squeeze(0)
        all_history_pixel_values.append(h_inputs['pixel_values'])
        all_history_image_grid_thw.append(h_inputs['image_grid_thw'])
        all_history_input_ids.append(h_input_ids)

        # Process similar trajectories as experience
        all_experience_input_ids = []
        all_experience_pixel_values = []
        all_experience_image_grid_thw = []
        
        for trajectory in similar_trajectories[:3]:  # Limit to 3 similar trajectories
            trajectory_text = ""
            trajectory_images = []
            
            actions = trajectory['actions']
            images = trajectory['images']
            
            for action, image_base64 in zip(actions, images):
                # Decode base64 image
                if isinstance(image_base64, dict) and image_base64.get('url', '').startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.get('url', '').split(',')[1])
                elif isinstance(image_base64, str) and image_base64.startswith('data:image/png;base64,'):
                    image_bytes = base64.b64decode(image_base64.split(',')[1])
                else:
                    image_bytes = base64.b64decode(image_base64)
                
                image = Image.open(BytesIO(image_bytes))
                trajectory_images.append(image)
                
                trajectory_text += f"{DEFAULT_IM_START_TOKEN}user\n{DEFAULT_IMAGE_TOKEN}{action}{DEFAULT_IM_END_TOKEN}\n"
            
            if trajectory_images:
                e_inputs = processor(text=[trajectory_text], images=trajectory_images, padding=False, return_tensors='pt')
                e_input_ids = e_inputs['input_ids'].squeeze(0)
                all_experience_pixel_values.append(e_inputs['pixel_values'])
                all_experience_image_grid_thw.append(e_inputs['image_grid_thw'])
            else:
                e_input_ids = processor.tokenizer(trajectory_text, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids'].squeeze(0)
            
            all_experience_input_ids.append(e_input_ids)

        # Concatenate all input_ids and labels
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        # Create data dictionary
        data_dict = dict(
            input_ids=input_ids,
            labels=labels,
            history_input_ids=all_history_input_ids,
            experience_input_ids=all_experience_input_ids,
        )

        # Add pixel values and grid info if images exist
        if all_pixel_values:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        if all_history_pixel_values:
            data_dict["history_pixel_values"] = all_history_pixel_values
            data_dict["history_image_grid_thw"] = all_history_image_grid_thw

        if all_experience_pixel_values:
            data_dict["experience_pixel_values"] = all_experience_pixel_values
            data_dict["experience_image_grid_thw"] = all_experience_image_grid_thw

        if len(all_second_gird) > 0:
            data_dict["second_per_grid_ts"] = all_second_gird

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        # EDIT: The batch size is 1
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        # Added for history
        batch_history_input_ids = []
        batch_history_pixel_values = []
        batch_history_image_thw = []
        # Added for experience
        batch_experience_input_ids = []
        batch_experience_pixel_values = []
        batch_experience_image_thw = []
        
        for example in examples:
            keys = example.keys()
            
            # Handle main input data
            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys and example["pixel_values"] is not None:
                # Only add main pixel values if they exist and are not None
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])
            
            # Handle history data (independent of main input)
            if "history_pixel_values" in keys:
                batch_history_pixel_values.append(example["history_pixel_values"])
                batch_history_image_thw.append(example["history_image_grid_thw"])
            
            # Handle experience data (independent of main input)
            if "experience_pixel_values" in keys:
                batch_experience_pixel_values.append(example["experience_pixel_values"])
                batch_experience_image_thw.append(example["experience_image_grid_thw"])
            
            # Always add main input_ids and labels
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            
            # Handle history and experience input_ids
            if "history_input_ids" in keys:
                batch_history_input_ids.append(example["history_input_ids"])
            if "experience_input_ids" in keys:
                batch_experience_input_ids.append(example["experience_input_ids"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.append(example["second_per_grid_ts"])
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='left', padding_value=self.pad_token_id
        )
        labels = pad_sequence(batch_label_ids, padding_side='left', padding_value=IGNORE_INDEX)
        attention_mask = input_ids != self.pad_token_id
        
        # Process history input_ids
        history_input_ids = []
        history_attention_mask = []
        if batch_history_input_ids:
            for history_input_ids_per_batch in batch_history_input_ids:
                if len(history_input_ids_per_batch) > 0:
                    padded_history = pad_sequence(history_input_ids_per_batch, padding_side='left', padding_value=self.pad_token_id)
                    history_input_ids.append(padded_history)
                    history_attention_mask.append(padded_history != self.pad_token_id)
        
        # Process experience input_ids
        experience_input_ids = []
        experience_attention_mask = []
        if batch_experience_input_ids:
            for experience_input_ids_per_batch in batch_experience_input_ids:
                if len(experience_input_ids_per_batch) > 0:
                    padded_experience = pad_sequence(experience_input_ids_per_batch, padding_side='left', padding_value=self.pad_token_id)
                    experience_input_ids.append(padded_experience)
                    experience_attention_mask.append(padded_experience != self.pad_token_id)
        
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        
        if history_input_ids:
            data_dict['history_input_ids'] = history_input_ids
            data_dict['history_attention_mask'] = history_attention_mask
            
        if experience_input_ids:
            data_dict['experience_input_ids'] = experience_input_ids
            data_dict['experience_attention_mask'] = experience_attention_mask

        # Handle main pixel values
        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw
        else:
            # If no main pixel values, explicitly set to None
            data_dict["pixel_values"] = None
            data_dict["image_grid_thw"] = None
            
        # Handle history pixel values (independent of main pixel values)
        if batch_history_pixel_values:
            history_pixel_values = [torch.cat(history_pixel_values_per_batch, dim=0) for history_pixel_values_per_batch in batch_history_pixel_values]
            history_image_thw = [torch.cat(history_image_thw_per_batch, dim=0) for history_image_thw_per_batch in batch_history_image_thw]
            data_dict["history_pixel_values"] = history_pixel_values
            data_dict["history_image_grid_thw"] = history_image_thw
            
        # Handle experience pixel values (independent of main pixel values)
        if batch_experience_pixel_values:
            experience_pixel_values = [torch.cat(experience_pixel_values_per_batch, dim=0) for experience_pixel_values_per_batch in batch_experience_pixel_values]
            experience_image_thw = [torch.cat(experience_image_thw_per_batch, dim=0) for experience_image_thw_per_batch in batch_experience_image_thw]
            data_dict["experience_pixel_values"] = experience_pixel_values
            data_dict["experience_image_grid_thw"] = experience_image_thw

        if len(batch_pixel_video_values) > 0:
            pixel_video_values = torch.cat(batch_pixel_video_values, dim=0)
            video_thw = torch.cat(batch_video_thw, dim=0)
            data_dict["pixel_values_videos"] = pixel_video_values
            data_dict["video_grid_thw"] = video_thw

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        return data_dict
    

def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r'\n?' + re.escape(LLAVA_VIDEO_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN

    return re.sub(pattern, replacement, input_string)

def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content[:5000],
        }
        transformed_data.append(transformed_entry)

    return transformed_data

def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path, processor=processor, data_args=data_args, model_id=model_id
    )
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)

    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)
