"""Training data collection utility for the GUI Agent"""

import json
import uuid
import copy
import base64
import gzip
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import re
from openai import OpenAI
from PIL import Image
import io

# Add the parent directory to the path to import modules
sys.path.insert(0, 'GUI-Agent')


class TrainingDataCollector:
    def __init__(self, output_dir: str = "training_data", enabled: bool = True):
        """
        Initialize the training data collector
        
        Args:
            output_dir: Directory to save training data
            enabled: Whether to enable data collection
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different trajectory types
        self.success_dir = self.output_dir / 'success'
        self.positive_dir = self.output_dir / 'positive'
        self.negative_dir = self.output_dir / 'negative'
        
        for dir_path in [self.success_dir, self.positive_dir, self.negative_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize OpenAI client for evaluation
        self.evaluation_client = OpenAI(
            base_url='http://localhost:8003/v1',
            api_key="EMPTY"
        )
        self.evaluation_model = "Zery/CUA_World_State_Model"
        
        # Conversation tracking
        self.session_id = f"session_{int(time.time())}"
        self.session_start = datetime.now()
        self.current_conversation_id = None
        self.conversation_start_time = None
        self.conversation_history = []
        self.conversation_task = None

    
    def start_conversation(self, conversation_id: str, task_description: Optional[str] = None):
        """Start a new conversation"""
        if not self.enabled:
            return
            
        self.current_conversation_id = conversation_id
        self.conversation_start_time = datetime.now()
        self.conversation_history = []
        self.conversation_task = task_description
        
        print(f"Started conversation: {conversation_id}")
    
    def add_conversation_round(self, messages: List[Dict[str, Any]], response: str):
        """Add a conversation round"""
        if not self.enabled or self.current_conversation_id is None:
            return
        messages = self.compress_base64_image_in_conversation(messages)
        response = self.clean_response(response)
        round_data = {
            "timestamp": datetime.now().isoformat(),
            "messages": messages,
            "response": response,
        }
        self.conversation_history.append(round_data)
    
    def evaluate_trajectory(self, conversation_data: Dict[str, Any], score: Optional[int] = None) -> Dict[str, Any]:
        """
        Evaluate a trajectory using the evaluation model
        
        Args:
            conversation_data: The conversation data to evaluate
            
        Returns:
            Evaluation results with correctness, error analysis, etc.
        """
        try:
            # Prepare the conversation for evaluation
            task_description = conversation_data.get('task_description', 'Unknown task')
            rounds = conversation_data.get('rounds', [])
            image_data = []
            responses = []
            for round_data in rounds:
                image_url = self.get_base64_image_from_conversation(round_data['messages'])
                if image_url:
                    image_data.append({"type": "image_url", "image_url": {"url": image_url}})
                    responses.append(round_data['response'])
            
            # Create evaluation prompt
            evaluation_prompt = (
    "I am evaluating the performance of a UI agent. The images provided are **sequential keyframes** that represent "
    "the full execution trajectory of the agent when attempting to follow a command. "
    f"These keyframes correspond to the instruction: **'{task_description}'**.\n\n"
    
    "Here are the actions of the agent:\n" + 
    "\n".join(f"Action {i+1}: {response}" for i, response in enumerate(responses)) +
    "\n\n"
    "Screenshot of the agent's actions are also provided.\n\n"
    
    "Please thoroughly analyze the sequence to assess the following aspects:\n"
    "1. **Correctness** -- Did the agent successfully complete the task as instructed?\n"
    "2. **Redundant Steps** -- Identify any unnecessary or repeated actions that do not contribute to the goal.\n"
    "3. **Optimization** -- Did the agent follow an efficient plan with a minimal number of steps?\n"
    "4. **First Error Step** -- If the execution is incorrect or sub-optimal, determine the index of the **first keyframe where a mistake occurred**.\n"
    "5. **Error Analysis** -- Provide a brief explanation of the mistake at that step.\n"
    "6. **Correct Action Suggestion** -- Explain what the agent **should have done instead** at the point of error.\n\n"

    "**Important Instructions:**\n"
    "- The agent may have made progress toward the goal, but unless the task is **fully and correctly completed**, you must set 'Correctness' to **False**.\n"
    "- Be cautious in determining success. Missing confirmation screens, skipped inputs, or wrong UI elements clicked all count as errors.\n"
    "- Carefully examine all UI changes, button interactions, text entries, and any visual feedback in the screenshots.\n"
    "- Clearly indicate **which exact steps are redundant** (starting from 1).\n\n"

    "Once you finish the analysis, return your evaluation in the following dictionary format (include your step-by-step reasoning **above** the result):\n\n"
    "<analysis process>\n"
    "<res_dict>{\n"
    "  \"Correctness\": True/False,\n"
    "  \"Redundant\": [step_num, ...],\n"
    "  \"Optimized\": True/False,\n"
    "  \"First_Error_Step\": step_num or None,\n"
    "  \"Error_Type\": \"brief description of the mistake\",\n"
    "  \"Correct_Action\": \"what should have been done instead\"\n"
    "}</res_dict>"
)
            messages = [{"role": "user", "content": [{"type": "text", "text": evaluation_prompt}] + image_data}]
            # Call the evaluation model
            response = self.evaluation_client.chat.completions.create(
                model=self.evaluation_model,
                messages=messages,
                temperature=1,
                max_tokens=2000
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Parse the response
            analysis_match = re.search(r'<analysis process>(.*?)</analysis>', evaluation_text, re.DOTALL)
            res_dict_match = re.search(r'<res_dict>(.*?)</res_dict>', evaluation_text, re.DOTALL)
            
            def parse_evaluation_text(text):
                pattern = r'"([^"]+)":\s*([^,\n]+|"[^"]+"|\[[^\]]+\])'
                matches = re.findall(pattern, text)
                result_dict = {'evaluation': {}}
                for key, value in matches:
                    try:
                        # For lists
                        if value.startswith('[') and value.endswith(']'):
                            value = json.loads(value)
                        # For booleans and numbers
                        elif value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.isdigit():
                            value = int(value)
                        # For strings (remove quotes)
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                    except:
                        pass
                    result_dict['evaluation'][key] = value
                return result_dict

            if analysis_match and res_dict_match:
                analysis = analysis_match.group(1).strip()
                res_dict_text = res_dict_match.group(1).strip()
                result_dict = parse_evaluation_text(evaluation_text)
                result_dict['analysis'] = analysis
                return result_dict
            else:
                result_dict = parse_evaluation_text(evaluation_text)
                result_dict['analysis'] = 'Evaluation failed'
                print("Could not parse evaluation response")
                if score == 1:
                    result_dict['evaluation']['Correctness'] = True
                else:
                    result_dict['evaluation']['Correctness'] = False
                return result_dict
            
        except Exception as e:
            print(f"Error during trajectory evaluation: {e}")
            traceback.print_exc()
            return {
                "analysis": f"Evaluation error: {str(e)}",
                "evaluation": {"Correctness": False, "Error_Type": f"Evaluation error: {str(e)}"},
                "raw_response": ""
            }

    def get_base64_image_from_conversation(self, messages):
        for msg in messages:
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if 'image_url' in item:
                        if 'url' in item['image_url']:
                            return item['image_url']['url']
        return None
    
    def compress_base64_image_in_conversation(self, messages):
        for msg in messages:
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if 'image_url' in item:
                        if 'url' in item['image_url'] and item['image_url']['url'].startswith('data:image/'):
                            item['image_url']['url'] = self.compress_base64_image(item['image_url']['url'])
        return messages
    
    def compress_base64_image(self, image_url):
        image_data = base64.b64decode(image_url.split('base64,')[1])
        image = Image.open(io.BytesIO(image_data))
        resized_image = image.resize((int(image.width / 1.5), int(image.height / 1.5)))
        buffer = io.BytesIO()
        resized_image.save(buffer, format='PNG')
        # Encode and return with proper prefix
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{encoded_image}"
    
    def clean_response(self, response):
        # Handle list case first
        if isinstance(response, list):
            response = response[0]
        # Handle dict case
        if isinstance(response, dict):
            return response.get('content', '')
        # String or any other type
        return response   
    
    def end_conversation(self, conversation_summary: Optional[Dict[str, Any]] = None, score: Optional[int] = None) -> str:
        """
        End the current conversation, evaluate it, and save it to appropriate folder
        
        Args:
            conversation_summary: Additional summary information about the conversation
            
        Returns:
            Path to the saved file
        """
        if not self.enabled or self.current_conversation_id is None:
            print("Training data collection is disabled or conversation ID is not set")
            return ""
        
        # Prepare the conversation data
        conversation_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "conversation_id": self.current_conversation_id,
            "conversation_start": self.conversation_start_time.isoformat() if self.conversation_start_time else None,
            "conversation_end": datetime.now().isoformat(),
            "task_description": getattr(self, 'conversation_task', None),
            "total_rounds": len(self.conversation_history),
            "rounds": self.conversation_history,
        }
        
        # Evaluate the trajectory
        evaluation_result = self.evaluate_trajectory(conversation_data, score)
        
        # Add evaluation to conversation data
        conversation_data["evaluation"] = evaluation_result
        print(evaluation_result)
        
        # Determine where to save based on correctness
        correctness = evaluation_result.get("evaluation", {}).get("Correctness", False)
        
        if correctness:
            # Save complete successful trajectory
            save_dir = self.success_dir
            filename = f"{self.current_conversation_id}.jsonl"
            trajectory_data = conversation_data
        else:
            # Split trajectory based on first error
            first_error_step = evaluation_result.get("evaluation", {}).get("First_Error_Step")
            if first_error_step and isinstance(first_error_step, int) and first_error_step > 2:
                # Save positive part (before first error)
                positive_data = conversation_data.copy()
                positive_data["rounds"] = conversation_data["rounds"][:first_error_step-1]
                positive_data["total_rounds"] = len(positive_data["rounds"])
                positive_data["split_type"] = "positive_part"
                positive_data["original_conversation_id"] = self.current_conversation_id
                
                # Save negative part (from first error onwards)
                negative_data = conversation_data.copy()
                negative_data["rounds"] = conversation_data["rounds"][first_error_step-1:]
                negative_data["total_rounds"] = len(negative_data["rounds"])
                negative_data["split_type"] = "negative_part"
                negative_data["original_conversation_id"] = self.current_conversation_id
                
                # Save both parts
                positive_filename = f"{self.current_conversation_id}.jsonl"
                negative_filename = f"{self.current_conversation_id}.jsonl"
                
                positive_path = self.positive_dir / positive_filename
                negative_path = self.negative_dir / negative_filename
                
                try:
                    with open(positive_path, 'w', encoding='utf-8') as f:
                        json.dump(positive_data, f, indent=2, ensure_ascii=False)
                    with open(negative_path, 'w', encoding='utf-8') as f:
                        json.dump(negative_data, f, indent=2, ensure_ascii=False)
                    
                    print(f"Split trajectory saved:")
                    print(f"  Positive part: {positive_path}")
                    print(f"  Negative part: {negative_path}")
                    
                    # Reset conversation tracking
                    self.current_conversation_id = None
                    self.conversation_history = []
                    self.conversation_start_time = None
                    self.conversation_task = None
                    
                    return str(positive_path)  # Return positive path as primary
                    
                except Exception as e:
                    print(f"Error saving split trajectory: {e}")
                    return ""
            else:
                # No need to save the trajectory
                print("No need to save the trajectory")
                return ""
        
        # Save single trajectory
        filepath = save_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
            
            # Log file size
            actual_size = filepath.stat().st_size if filepath.exists() else 0
            size_mb = actual_size / (1024 * 1024)
            print(f"Trajectory saved: {filepath} ({size_mb:.2f} MB)")
            
            # Reset conversation tracking
            self.current_conversation_id = None
            self.conversation_history = []
            self.conversation_start_time = None
            self.conversation_task = None
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving trajectory: {e}")
            traceback.print_exc()
            return ""
    
    def disable(self):
        """Disable data collection"""
        self.enabled = False
        print("Training data collection disabled")
    
    def enable(self):
        """Enable data collection"""
        self.enabled = True
        print(f"Training data collection enabled. Output directory: {self.output_dir}")


def get_collector() -> TrainingDataCollector:
    """Get the global training data collector instance"""
    global _global_collector
    if _global_collector is None:
        _global_collector = TrainingDataCollector()
    return _global_collector

def set_collector(collector: TrainingDataCollector):
    """Set the global training data collector instance"""
    global _global_collector
    _global_collector = collector 