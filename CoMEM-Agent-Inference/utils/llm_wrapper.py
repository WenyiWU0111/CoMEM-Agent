"""LLM wrapper to capture actual model inputs including function prompts"""

import copy
from typing import Dict, List, Any, Optional, Iterator, Union
from utils.training_data_collector import get_collector


class LLMWrapper:
    """Wrapper around LLM to capture actual model inputs"""
    
    def __init__(self, llm):
        """
        Initialize the LLM wrapper
        
        Args:
            llm: The underlying LLM instance
        """
        # Initialize the base class with the LLM's config
        self.llm = llm
        self.collector = get_collector()
        
        # Copy important attributes from the wrapped LLM
        self.model = getattr(llm, 'model', '')
        self.model_type = getattr(llm, 'model_type', '')
        self.generate_cfg = getattr(llm, 'generate_cfg', {})
        
        # Store last response and usage info for token tracking
        self.last_response = None
        self.last_usage = None
    
    def chat(self, 
             messages: List[Union[Any, Dict]], 
             stream: bool = True,
             delta_stream: bool = False,
             extra_generate_cfg: Optional[Dict] = None,
             **kwargs) -> Union[List[Any], List[Dict], Iterator[List[Any]], Iterator[List[Dict]]]:
        """
        Chat with the LLM and capture the actual input sent to the model
        
        Args:
            messages: Input messages
            functions: Available functions
            stream: Whether to stream the response
            delta_stream: Whether to use delta streaming
            extra_generate_cfg: Extra generation configuration
            **kwargs: Additional arguments
            
        Returns:
            LLM response
        """
        # Call the original LLM with all parameters
        response = self.llm.chat(messages=messages, **kwargs)
        
        # Store the last response and usage info for token tracking
        self.last_response = response
        
        # Manually count tokens from input messages using Qwen2.5-VL tokenizer
        # try:
        from transformers import AutoProcessor
        import base64
        from PIL import Image
        import io
        
        # # Load Qwen2.5-VL tokenizer
        # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        
        # Count tokens in all messages
        total_tokens = 0
        for message in messages:
            if isinstance(message, dict):
                content = message.get('content', '')
                if isinstance(content, list):
                    # Handle multimodal content (text + images)
                    text_parts = []
                    image_parts = []
                    
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'image_url':
                                image_url = item.get('image_url', {}).get('url', '')
                                if image_url.startswith('data:image/'):
                                    # Decode base64 image
                                    try:
                                        image_data = base64.b64decode(image_url.split(',')[1])
                                        image = Image.open(io.BytesIO(image_data))
                                        image_parts.append(image)
                                    except Exception as e:
                                        print(f"Warning: Could not decode image: {e}")
                                else:
                                    # URL image - try to load
                                    try:
                                        image = Image.open(image_url)
                                        image_parts.append(image)
                                    except Exception as e:
                                        print(f"Warning: Could not load image from URL: {e}")
                    
                    # Tokenize text and images together
                    text = ' '.join(text_parts)
                    # inputs = processor(text=text, images=image_parts, return_tensors="pt")
                    # tokens = inputs.input_ids[0]
                    # total_tokens += len(tokens)
            else:
                # # Simple text message
                # inputs = processor(text=str(message), return_tensors="pt")
                # tokens = inputs.input_ids[0]
                # total_tokens += len(tokens)
                pass
        
        self.last_usage = {'prompt_tokens': total_tokens}
        print(f"Qwen2.5-VL tokenizer counted tokens: {total_tokens}")
        
        return response
    
        # except Exception as e:
        #     print(f"Warning: Could not count tokens with Qwen2.5-VL tokenizer: {e}")
        #     self.last_usage = None
    
    def save_conversation(self, messages, response):
        # Capture the actual model input if collector is enabled
        if self.collector and self.collector.enabled:
            try:
                # Add to conversation if one is active, otherwise collect as single interaction
                if hasattr(self.collector, 'current_conversation_id') and self.collector.current_conversation_id:
                    self.collector.add_conversation_round(
                        messages=messages,
                        response=response,
                    )
            except Exception as e:
                print(f"Warning: Could not save conversation: {e}")
        
        
    
    def __getattr__(self, name):
        """Delegate all other attributes to the underlying LLM"""
        return getattr(self.llm, name)


def wrap_llm(llm) -> LLMWrapper:
    """
    Wrap an LLM instance to capture actual model inputs
    
    Args:
        llm: The LLM instance to wrap
        
    Returns:
        Wrapped LLM instance
    """
    return LLMWrapper(llm) 