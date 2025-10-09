import os
import json
import numpy as np
import faiss
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.help_functions import CLIPTextSimilarity, CLIPMultimodalSimilarity
from actions.help_functions import parse_action_json
from PIL import Image
import base64
from io import BytesIO
from datetime import datetime
import logging
import re
from typing import Optional, Dict

LOG_PATH = "memory"
clip_similarity = CLIPTextSimilarity()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.FileHandler(LOG_PATH + '/memory.log'))


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

class Memory:
    """
    A new Memory class that directly uses all jsonl files from training_data
    and creates a single pool of memories with embeddings for better generalization.
    """
    
    def __init__(self, training_data_path="training_data", agent=None, faiss_index_path=None, multimodal=False, bank_size=None):
        self.training_data_path = training_data_path
        self.multimodal = multimodal
        self.selected_conversations = None
        self.agent = agent
        if multimodal:
            self.clip_similarity = CLIPMultimodalSimilarity()
        else:
            self.clip_similarity = CLIPTextSimilarity()
            
        self.memories = []  # Single pool of all memories
        self.embeddings = None  # Embedding matrix for all memories
        self.faiss_index = None  # FAISS index for fast similarity search
        self.bank_size = bank_size

        if faiss_index_path is None:
            print('Generating new memory index...')
            self._load_all_conversations()
            self._create_faiss_index()
            os.makedirs(f"memory_index", exist_ok=True)
            self.save_index(f"memory_index/{'multimodal' if multimodal else 'text'}_{self.faiss_index.ntotal}")
        else:
            print(f'Loading memory index from {faiss_index_path}...')
            self.load_index(faiss_index_path)
    
    def _load_all_conversations(self):
        """Load all conversations from the training data directory into a single pool."""
        print("Loading all conversations from training data...")
        
        # Get all dataset folders
        dataset_folders = [f for f in os.listdir(self.training_data_path) 
                          if os.path.isdir(os.path.join(self.training_data_path, f))]
        seen_files = set()
        total_conversations = 0
        for dataset in dataset_folders:
            # if 'conversation' in dataset:
            #     continue
            dataset_path = os.path.join(self.training_data_path, dataset)
            
            # Get all domain folders within each dataset
            domain_folders = [f for f in os.listdir(dataset_path) 
                             if os.path.isdir(os.path.join(dataset_path, f))]
            
            for domain in domain_folders:
                domain_path = os.path.join(dataset_path, domain)
                
                # Get all model folders (like qwen2.5-vl)
                model_folders = [f for f in os.listdir(domain_path) 
                                if os.path.isdir(os.path.join(domain_path, f))]
                
                for model in model_folders:
                    if model != 'qwen2.5-vl-32b':
                        continue
                    model_seen_files = set()
                    model_path = os.path.join(domain_path, model)
                    
                    # Get all test folders or direct jsonl files
                    items = os.listdir(model_path)
                    
                    # Handle different structures: some have test folders, others have direct jsonl files
                    jsonl_files = []
                    for item in items:
                        item_path = os.path.join(model_path, item, 'success')
                        if os.path.isdir(item_path):
                            # Check if this directory contains jsonl files
                            sub_items = os.listdir(item_path)
                            for sub_item in sub_items:
                                sub_item_path = os.path.join(item_path, sub_item)
                                if not os.path.isfile(sub_item_path):
                                    continue
                                if sub_item.endswith('.jsonl'):
                                    if sub_item_path not in model_seen_files:
                                        jsonl_files.append(sub_item_path)
                                        model_seen_files.add(sub_item_path)
                    
                    # Load conversations from jsonl files
                    for jsonl_file in jsonl_files:
                        try:
                            with open(jsonl_file, 'r') as f:
                                memory_file = json.load(f)
                                task_description = memory_file['task_description']
                                total_rounds = memory_file['total_rounds']
                                if total_rounds < 3 or total_rounds >= 15:
                                    logger.info(f"Skipping {jsonl_file} because it has {total_rounds} rounds")
                                    continue
                                if task_description == '':
                                    logger.info(f"Skipping {jsonl_file} because task description is empty")
                                    continue
                                prefixed_query = f"{dataset}_{domain}: {task_description}"
                                if prefixed_query in seen_files:
                                    logger.info(f"Skipping {jsonl_file} because it has already been seen")
                                    continue
                                conversation_list = memory_file['rounds']
                                # responses_list = [conversation['response'] for conversation in conversation_list]
                                base64_image = self._extract_base64_image(conversation_list[0])
                                # actual_actions = []
                                # previous_action_name, previous_action_reasoning = None, None
                                # for response in responses_list:
                                #     action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response)
                                #     if action_json:
                                #         if current_action_name == previous_action_name:
                                #             continue
                                #         else:
                                #             actual_actions.append(action_json)
                                #             previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                                #     else:
                                #         # print(f"Error parsing action: {response}")
                                #         continue
                                # if len(actual_actions) < 3:
                                #     logger.info(f"Skipping {jsonl_file} because it has {len(actual_actions)} actions")
                                #     continue
                                
                                self.memories.append({
                                    'file_path': jsonl_file,
                                    'task_description': task_description,
                                    'prefixed_query': prefixed_query,
                                    'dataset': dataset,
                                    'domain': domain,
                                    'base64_image': base64_image
                                })
                                seen_files.add(prefixed_query)
                                total_conversations += 1
                                        
                        
                        except Exception as e:
                            logger.info(f"Error loading {jsonl_file}: {e}")
                            continue
        
        print(f"Total conversations loaded: {len(self.memories)}")
    
    def _extract_base64_image(self, data):
        """Extract base64 image from conversation data."""
        try:
            # Check if data has messages
            if 'messages' in data:
                messages = data['messages']
                for msg in messages:
                    if isinstance(msg.get('content'), list):
                        for item in msg['content']:
                            if item.get('type') == 'image_url':
                                return item['image_url']['url']
            return None
        except Exception as e:
            print(f"Error extracting base64 image: {e}")
            return None
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search."""
        print("Creating FAISS index for all memories...")
        if not self.memories:
            print("No memories to create FAISS index for")
            return
        
        # Extract all prefixed queries and base64 images
        prefixed_queries = [memory['prefixed_query'] for memory in self.memories]
        base64_images = [memory.get('base64_image') for memory in self.memories]
        
        # Create embeddings using CLIP
        if self.multimodal:
            # For multimodal, we always create multimodal embeddings
            # Use None for missing images to maintain consistent dimensions
            self.embeddings = self.clip_similarity.get_multimodal_embeddings(prefixed_queries, base64_images)
        else:
            self.embeddings = self.clip_similarity.get_text_embeddings(prefixed_queries)
        
        logger.info(f"Created embeddings matrix with shape: {self.embeddings.shape}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to the index
        self.faiss_index.add(self.embeddings.astype('float32'))
        
        print(f"Created FAISS index with {self.faiss_index.ntotal} vectors")
    
    def retrieve_similar_conversations(self, current_question, current_image=None, model=None, similar_num=3):
        """
        Retrieve similar conversations based on text and/or image similarity from the single memory pool using FAISS.
        
        Args:
            current_question: The current query/question
            current_image: Optional base64 encoded image for multimodal search
            similar_num: Number of similar conversations to retrieve
        
        Returns:
            List of selected conversation file paths
        """
        if not self.memories or self.faiss_index is None:
            logger.info("No memories available for retrieval")
            return []
        if model is not None:
            current_question = f"{model}: {current_question}"
        # Get embedding for current question and image
        if self.multimodal:
            if current_image is not None:
                current_embedding = self.clip_similarity.get_multimodal_embeddings([current_question], [current_image])
            else:
                # For multimodal mode with no image, we need to create embeddings with the same dimension
                # as the stored embeddings (which are text+image concatenated)
                text_embedding = self.clip_similarity.get_text_embeddings([current_question])
                # Create zero embeddings for the image part to match the stored dimension
                zero_image_embedding = np.zeros_like(text_embedding)
                current_embedding = np.concatenate([text_embedding, zero_image_embedding], axis=1)
        else:
            current_embedding = self.clip_similarity.get_text_embeddings([current_question])
            zero_image_embedding = np.zeros_like(current_embedding)
            current_embedding = np.concatenate([current_embedding, zero_image_embedding], axis=1)
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(current_embedding)
        
        # Search using FAISS
        similarities, indices = self.faiss_index.search(
            current_embedding.astype('float32'), similar_num
        )
        
        selected_conversations = []
        for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                if current_question.split(':')[-1].strip() in self.memories[idx]['prefixed_query']:
                    continue
                selected_conversations.append(self.memories[idx]['file_path'])
                logger.info(f"Score: {score:.4f} - {self.memories[idx]['prefixed_query']}")
        
        return selected_conversations
    
    def parse_action_from_response(self, response):
        """
        Parse action from response with fallback to LLM parsing.
        
        Args:
            response: The response to parse
        
        Returns:
            tuple: (action_json, current_action_name, current_action_reasoning) or (None, None, None) if parsing fails
        """
        try:
            if isinstance(response, list):
                response = response[0]
            if isinstance(response, dict) and 'content' in response:
                response = response['content']
            action_json = parse_action_json(response).get('function_call', {})
            
            if 'name' in action_json:
                current_action_name = action_json['name']
                current_action_reasoning = action_json['arguments']['reasoning']
            elif 'action' in action_json:
                current_action_name = action_json['action']
                current_action_reasoning = action_json['reasoning']
            elif 'action_type' in action_json:
                current_action_name = action_json['action_type']
                current_action_reasoning = action_json['reasoning']
            elif 'type' in action_json:
                current_action_name = action_json['type']
                current_action_reasoning = action_json['reasoning']
            elif isinstance(list(action_json.values())[0], dict):
                current_action_name = list(action_json.keys())[0]
                current_action_reasoning = list(action_json.values())[0]['reasoning']
            else:
                print(f"Error: {action_json} has no name, action, or action_type")
                return None, None, None
            
            action_json['name'] = current_action_name
            action_json['arguments'] = {'reasoning': current_action_reasoning}
            
            return action_json, current_action_name, current_action_reasoning
        
        except:
            try:
                action_json = self.agent._parse_natural_language_with_llm(response, pure_text=True)
                current_action_name = action_json['name']
                current_action_reasoning = action_json['arguments']['reasoning']
                
                return action_json, current_action_name, current_action_reasoning
            except:
                logger.info(f"Error parsing action: {response}")
                return None, None, None

    def construct_experience_memory(self, current_question, agent, current_image=None, dataset=None, domain=None, similar_num=3):
        """
        Construct experience memory from similar conversations.
        
        Args:
            current_question: The current query/question
            agent: The agent instance for parsing actions
            current_image: Optional base64 encoded image for multimodal search
            dataset: Optional dataset filter
            domain: Optional domain filter
            similar_num: Number of similar conversations to use
        
        Returns:
            Formatted experience memory string
        """
        current_question = f"{dataset}_{domain}: {current_question}" if dataset and domain else current_question
        selected_conversations = self.retrieve_similar_conversations(
            current_question=current_question, current_image=current_image, similar_num=similar_num + 5
        )
        self.selected_conversations = selected_conversations
        
        desc_list = []
        action_texts_list = []
        images_list = []
        file_id_list = []
        
        for conversation_file in selected_conversations:
            try:
                with open(conversation_file, 'r') as f:
                    memory_file = json.load(f)
                    task_description = memory_file['task_description']
                    if task_description == '':
                        logger.info(f"Task description is empty for {conversation_file}")
                        continue
                    conversation_list = memory_file['rounds']
                    responses_list = [conversation['response'] for conversation in conversation_list]
                    images_list_per_conversation = []
                    for conversation in conversation_list:
                        image = self._extract_base64_image(conversation)
                        images_list_per_conversation.append(image)
                if len(images_list_per_conversation) != len(responses_list):
                    print(f"Error: {conversation_file} has {len(images_list_per_conversation)} images and {len(responses_list)} responses")
                    continue
                
                actual_actions = []
                actual_images = []
                previous_action_name, previous_action_reasoning = None, None
                
                for response, image in zip(responses_list, images_list_per_conversation):
                    if isinstance(response, list):
                        response = response[0]
                    try:
                        if isinstance(response, dict) and 'content' in response:
                            response = response['content']
                    except:
                        print(f"Error response: {response}")
                    
                    action_json, current_action_name, current_action_reasoning = self.parse_action_from_response(response)
                    
                    if action_json:
                        if current_action_name == previous_action_name: #and current_action_reasoning == previous_action_reasoning
                            continue
                        else:
                            actual_actions.append(action_json)
                            actual_images.append(image)
                            previous_action_name, previous_action_reasoning = current_action_name, current_action_reasoning
                    else:
                        print(f"Error parsing action: {response}")
                        continue
                
                if len(actual_actions) >= 10:
                    actual_actions = actual_actions[::2]
                    actual_images = actual_images[::2]
                actions_desc = f"EXAMPLE: {task_description}\n"
                for action in actual_actions:
                        actions_desc += f"{action['name']}: {action['arguments']['reasoning']}\n"
                
                desc_list.append(actions_desc)
                action_texts_list.append(actual_actions)
                images_list.append(actual_images)
                file_id_list.append(conversation_file.split('/')[-1].split('.')[0])
                    
            except Exception as e:
                print(f"Error processing {conversation_file}: {e}")
        
        if len(action_texts_list) > 0:
            return '\n'.join(desc_list[:similar_num]), action_texts_list[:similar_num], images_list[:similar_num], file_id_list[:similar_num]
        else:
            return "", [], [], []
                    
    
    def get_available_datasets_and_domains(self):
        """Get list of available datasets and domains."""
        result = {}
        for memory in self.memories:
            dataset = memory['dataset']
            domain = memory['domain']
            if dataset not in result:
                result[dataset] = []
            if domain not in result[dataset]:
                result[dataset].append(domain)
        return result
    
    def save_index(self, filepath):
        """Save the FAISS index, embeddings, and memory data to disk."""
        if self.faiss_index is None:
            print("No FAISS index to save")
            return
        
        # Save FAISS index
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        faiss.write_index(self.faiss_index, f"{filepath}.faiss")
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(f"{filepath}.embeddings.npy", self.embeddings)
        
        # Save memory data
        memory_data = {
            'memories': self.memories,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None
        }
        
        with open(f"{filepath}.json", 'w') as f:
            json.dump(memory_data, f, indent=2)
        
        print(f"Saved FAISS index, embeddings, and memory data to {filepath}")
    
    def load_index(self, filepath):
        """Load the FAISS index, embeddings, and memory data from disk."""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index(f"{filepath}.faiss")
            
            # Load embeddings
            embeddings_path = f"{filepath}.embeddings.npy"
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                print(f"Loaded embeddings with shape: {self.embeddings.shape}")
                if self.bank_size is not None:
                    self.embeddings = self.embeddings[:self.bank_size]
                    dimension = self.embeddings.shape[1]
                    new_index = faiss.IndexFlatIP(dimension)
                    new_index.add(self.embeddings.astype('float32'))
                    self.faiss_index = new_index
                    print('Cut embeddings to new size: ', self.embeddings.shape)
            else:
                print("Embeddings file not found, reconstructing from FAISS index...")
                self.embeddings = self.faiss_index.reconstruct_n(0, self.faiss_index.ntotal)
                if self.bank_size is not None:
                    self.embeddings = self.embeddings[:self.bank_size]
            # Load memory data
            with open(f"{filepath}.json", 'r') as f:
                memory_data = json.load(f)
            
            self.memories = memory_data['memories']
            
            print(f"Loaded FAISS index and memory data from {filepath}")
            print(f"Index contains {self.faiss_index.ntotal} vectors")
            print(f"Loaded {len(self.memories)} memories")
            
        except Exception as e:
            print(f"Error loading index from {filepath}: {e}")
            print("Falling back to creating new index...")
            self._load_all_conversations()
            self._create_faiss_index()


    def retrieve_similar_conversations_with_filter(self, current_question, current_image=None, dataset=None, domain=None, similar_num=3):
        """
        Retrieve similar conversations with optional dataset/domain filtering.
        This method filters the memory pool before similarity search.
        
        Args:
            current_question: The current query/question
            current_image: Optional base64 encoded image for multimodal search
            dataset: Optional dataset filter
            domain: Optional domain filter
            similar_num: Number of similar conversations to retrieve
        
        Returns:
            List of selected conversation file paths
        """
        if not self.memories or self.faiss_index is None:
            print("No memories available for retrieval")
            return []
        
        # Filter memories based on dataset and domain if specified
        filtered_memories = []
        filtered_indices = []
        
        for i, memory in enumerate(self.memories):
            if dataset and memory['dataset'] != dataset:
                continue
            if domain and memory['domain'] != domain:
                continue
            filtered_memories.append(memory)
            filtered_indices.append(i)
        
        if not filtered_memories:
            print("No memories found for the specified criteria")
            return []
        
        # Get embeddings for filtered memories
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(filtered_embeddings)
        
        # Create a temporary FAISS index for filtered embeddings
        dimension = filtered_embeddings.shape[1]
        temp_index = faiss.IndexFlatIP(dimension)
        temp_index.add(filtered_embeddings.astype('float32'))
        
        # Get embedding for current question and image
        if self.multimodal:
            if current_image is not None:
                current_embedding = self.clip_similarity.get_multimodal_embeddings([current_question], [current_image])
            else:
                # For multimodal mode with no image, we need to create embeddings with the same dimension
                # as the stored embeddings (which are text+image concatenated)
                text_embedding = self.clip_similarity.get_text_embeddings([current_question])
                # Create zero embeddings for the image part to match the stored dimension
                zero_image_embedding = np.zeros_like(text_embedding)
                current_embedding = np.concatenate([text_embedding, zero_image_embedding], axis=1)
        else:
            current_embedding = self.clip_similarity.get_text_embeddings([current_question])
        
        # Normalize embedding for cosine similarity
        faiss.normalize_L2(current_embedding)
        
        # Search using temporary FAISS index
        similarities, indices = temp_index.search(
            current_embedding.astype('float32'), similar_num
        )
        
        selected_conversations = []
        for i, (score, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1:  # FAISS returns -1 for invalid indices
                memory_idx = filtered_indices[idx]
                selected_conversations.append(self.memories[memory_idx]['file_path'])
                print(f"Score: {score:.4f} - {self.memories[memory_idx]['prefixed_query']}")
        
        return selected_conversations

    