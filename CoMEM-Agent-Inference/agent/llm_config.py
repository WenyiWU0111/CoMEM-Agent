"""LLM configuration for different model types"""
import argparse
from typing import Dict, List
import base64
from openai import OpenAI
from transformers import AutoProcessor
import torch
import sys
from PIL import Image
from io import BytesIO
from openai.types.chat import ChatCompletionMessage

class DirectVLLMModel:
    """Direct vLLM model wrapper that can be used without qwen_agent"""
    
    def __init__(self, model_name: str, server_url: str, api_key: str = "EMPTY", **kwargs):
        self.model_name = model_name
        self.server_url = server_url
        self.api_key = api_key
        self.client = OpenAI(
            base_url=server_url,
            api_key=api_key
        )
        self.temperature = kwargs.get('temperature', 0.2)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = kwargs.get('max_tokens', 2048)
    
    def chat(self, messages: List[Dict], stream: bool = False, **kwargs):
        """Chat with the model using simplified message format"""
        # Prepare function calling parameters
        call_params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.temperature),
            "top_p": kwargs.get('top_p', self.top_p),
            "max_tokens": kwargs.get('max_tokens', self.max_tokens),
        }
        
        # Call the model
        response = self.client.chat.completions.create(**call_params)
        
        if stream:
            return response, None, None
        else:
            return response.choices[0].message, None, None

class DirectTransformersModel:
    """Direct Transformers model wrapper for Qwen2.5-VL with experience handling"""
    
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.1)
        self.top_p = kwargs.get('top_p', 0.9)
        self.max_tokens = 10**4
        self.checkpoint_path = kwargs.get('checkpoint_path', model_name)
        
        # Load processor and tokenizer
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)
        self.tokenizer = self.processor.tokenizer
        
        # Import the custom model class
        path = 'CoMEM-Agent/CoMEM-Agent-train'
        sys.path.append(path)
        from src_agent.training.qwenVL_inference import Qwen2_5_VLForConditionalGeneration_new
        self.model = Qwen2_5_VLForConditionalGeneration_new.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
    def process_vision_info(self, conversation):
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
    
    def knowledge_processor_vlm(self, processor, inputs, texts=None, images=None, tokenizer=None, formatted_prompt=None):
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
    
    def generate_response_with_experience(self, image=None, prompt=None, experience_texts=None, experience_images=None, file_id_list=None, conversation=None, experience_embedding=None):
        """Generate response with experience texts and images"""
        
        if not conversation:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image}
                    ],
                }
            ]
        
        formatted_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print('formatted_prompt:', formatted_prompt)
        image_inputs = self.process_vision_info(conversation)
        # print('image_number:', len(image_inputs))
        
        inputs = self.processor(
            text=[formatted_prompt],
            images=image_inputs,
            return_tensors="pt",
        ).to("cuda")
        file_id_list = None
        if file_id_list is not None:
            inputs['file_id_list'] = file_id_list
            inputs_with_experience = inputs
        else:
            # Process experience information
            inputs_with_experience = self.knowledge_processor_vlm(
                processor=self.processor,
                inputs=inputs,
                texts=experience_texts,
                images=experience_images,
                tokenizer=self.tokenizer,
                formatted_prompt=formatted_prompt
            ).to("cuda")
        
        generated_ids = self.model.generate(
            **inputs_with_experience, 
            max_new_tokens=self.max_tokens,
            use_cache=True, 
            temperature=self.temperature,
            top_p=self.top_p,
        )
        
        print('generated_ids', generated_ids)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs_with_experience.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        output_text = output_text[0]
        return output_text
    
    def chat(self, messages: List[Dict], stream: bool = False, 
             experience_texts=None, experience_images=None, file_id_list=None):
        """Chat with the model using transformers with experience support"""
        if stream:
            raise NotImplementedError("Streaming not yet implemented for transformers models")
        # Check if experience data is provided
        has_experience = False
        if experience_texts is not None:
            # Check if any experience text is not empty
            has_experience = any(len(text_list) > 0 for text_list in experience_texts)
        if experience_images is not None:
            # Check if any experience image list is not empty
            has_experience = any(len(img_list) > 0 for img_list in experience_images)

        if not has_experience:
            print("No experience data provided, falling back to DirectVLLMModel...")
            # Fall back to DirectVLLMModel when no experience data
            vllm_model = DirectVLLMModel(
                model_name='Qwen/Qwen2.5-VL-7B-Instruct',
                server_url='http://localhost:8000/v1',
                api_key="EMPTY",
                temperature=0.2,
                top_p=0.9,
                max_tokens=self.max_tokens
            )
            return vllm_model.chat(messages, stream=False)
        
        else:
            print("Generating response with experience...")
            # Generate response with experience
            response_text = self.generate_response_with_experience(
                experience_texts=experience_texts,
                experience_images=experience_images,
                file_id_list=file_id_list,
                conversation=messages
            )
            
            # Create OpenAI-style response
            return ChatCompletionMessage(
                role="assistant",
                content=response_text,
                function_call=None,
                tool_calls=None
            ), None, None


def create_direct_vllm_model(args: argparse.Namespace, model_name: str = None) -> DirectVLLMModel:
    """Create a direct vLLM model instance"""
    if model_name is None:
        model_name = args.model
    
    model_name_map = {
        'qwen2.5-vl': 'Qwen/Qwen2.5-VL-7B-Instruct',
        'qwen2-vl': 'Qwen/Qwen2-VL-7B-Instruct',
        'ui-tars': 'ByteDance-Seed/UI-TARS-1.5-7B',
        'websight': 'WenyiWU0111//websight-7B_combined',
        'cogagent': 'zai-org/cogagent-9b-20241220',
        'qwen2.5-vl-32b': 'Qwen/Qwen2.5-VL-32B-Instruct',
        'fuyu': 'adept/fuyu-8b',
        'gemini': 'google/gemini-2.5-pro',
        'claude': 'anthropic/claude-sonnet-4',
        'gpt-4o': 'openai/gpt-4o',
    }
    model_server_map = {
        'qwen2.5-vl': 'http://localhost:8000/v1',
        'qwen2-vl': 'http://localhost:8002/v1',
        'websight': 'http://localhost:8002/v1',
        'ui-tars': 'http://localhost:8001/v1',
        'cogagent': 'http://localhost:8002/v1',
        'fuyu': 'http://localhost:8002/v1',
        'qwen2.5-vl-32b': 'http://localhost:8004/v1',
        'gemini': 'https://openrouter.ai/api/v1',
        'claude': 'https://openrouter.ai/api/v1',
        'gpt-4o': 'https://openrouter.ai/api/v1',
    }

    model_name_ = model_name_map.get(model_name, model_name)
    server_url = model_server_map[model_name]
    api_key = args.get('open_router_api_key', 'EMPTY')
    print('model_name', model_name_)
    print('server_url', server_url)
    print('api_key', api_key)
    
    return DirectVLLMModel(
        model_name=model_name_,
        server_url=server_url,
        api_key=api_key,
        temperature=0.2,
        top_p=0.9,
        max_tokens=1024,
    )


def create_direct_transformers_model(args: argparse.Namespace, model_name: str = None) -> DirectTransformersModel:
    """Create a direct Transformers model instance"""
    model_name_map = {
        'agent-qformer': 'WenyiWU0111/lora_qformer_test_V4-700_merged',
        'ui-tars': 'WenyiWU0111/lora_qformer_uitars_test_V1-400_merged'
    }
    if model_name is None:
        model_name = model_name_map.get(args.model, args.model)
    else:
        model_name = model_name_map.get(model_name, model_name)
    
    return DirectTransformersModel(
        model_name=model_name,
        checkpoint_path=args.checkpoint_path if hasattr(args, 'checkpoint_path') else model_name,
        temperature=0.1,
        top_p=0.001,
    )


def create_direct_model(args: argparse.Namespace):
    """Create a direct model instance based on model type"""
    if args.use_continuous_memory:
        return create_direct_transformers_model(args)
    else:
        # Default to vLLM
        return create_direct_vllm_model(args)


def load_grounding_model_vllm(args: argparse.Namespace):
    """
    Load grounding model using vLLM server with OpenAI client.
    
    Args:
        args: Arguments object
        
    Returns:
        Grounding model client
    """
    # Create client with custom base URL pointing to your vLLM server
    grounding_model = create_direct_vllm_model(args, model_name='ui-tars')
    return grounding_model

def load_tool_llm(args: argparse.Namespace) -> DirectVLLMModel:
    """Load tool LLM"""
    tool_model = create_direct_vllm_model(args, model_name='qwen2.5-vl')
    return tool_model

