"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
import base64
import io
from typing import List

from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from webvoyager_evaluation.evaluator import LLMEvaluator
from utils.early_stop import early_stop
"""Test runner for the GUI Agent"""
import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
import base64
import io


from browser_env import (
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from utils.early_stop import early_stop
from utils.help_functions import save_scores_to_json
from agent.llm_config import load_tool_llm


class TestRunner:
    """Handles the main test execution loop"""
    
    def __init__(self, args: argparse.Namespace, agent):
        self.args = args
        self.agent = agent
        self.logger = logging.getLogger("logger")
        # Initialize environment
        self.env = ScriptBrowserEnv(
            headless=True,
            slow_mo=args.slow_mo,
            viewport_size={
                "width": args.viewport_width,
                "height": args.viewport_height,
            },
            save_trace_enabled=args.save_trace_enabled,
            sleep_after_execution=args.sleep_after_execution,
            args=args,  # Pass args to the environment
        )
        evaluate_model = load_tool_llm(self.args)
        self.evaluator = LLMEvaluator(vllm_client=evaluate_model)
        
    def run(self, config_file_list: list[str]):
        """Run the main test loop"""
        # Process each config file
        for config_file in config_file_list:
            self._process_config_file(config_file)
        # Close environment
        self.env.close()
    
    def _process_config_file(self, config_file: str):
        """Process a single config file"""
        action_list = []

        render_helper = RenderHelper(config_file, self.args.result_dir)

        # Get intent and task info
        with open(config_file) as f:
            _c = json.load(f)
            intent = _c["intent"]    
            task_id = _c["task_id"]
            site = _c["site"]
        
        episode_id = f"{site}_{task_id}"

        numbers = re.findall(r'\d+', config_file)
        self.args.task_cnt = int(numbers[0]) if numbers else None
        self.args.hop_cnt = 0
        
        self.logger.info(f"[Config file]: {config_file}")
        self.logger.info(f"[Intent]: {intent}")
        
        self.agent.reset(config_file)
        self.agent.current_step = 0
        trajectory: Trajectory = []
        
        # Environment reset
        obs, info = self.env.reset(
            options={"config_file": config_file}, 
        )
        current_url = info["page"].url
        state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
        trajectory.append(state_info)
        print("CURRENT: ", current_url)
        if 'about:blank' in current_url or info["is_blocked"]:
            self.logger.info(f"[Result] (Cannot navigate to {_c['start_url']}) {config_file}")
            return
        
        meta_data = {"action_history": [],
                     "response_history": []}
        
        print("config_file: ", config_file)
        
        # Information accumulation storage
        sub_query_answers = []
        
        # Start conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            collector = self.agent.training_collector
            if collector and collector.enabled:
                # Create conversation ID from task info
                conversation_id = f"{site}_{config_file.split('/')[-1].split('.')[0]}".replace(' ', '_')
                collector.start_conversation(
                    conversation_id=conversation_id,
                    task_description=intent
                )
                self.logger.info(f"Started conversation collection for task: {conversation_id}")
        
        if self.args.subtask:
            # Implement subtask decomposition with LLM
            intent_list = self._decompose_task_into_subtasks(intent)
            self.logger.info(f"Task decomposed into {len(intent_list)} subtasks")
        else:
            intent_list = [intent]

        # Process each sub-query sequentially
        for sub_query_idx, current_intent in enumerate(intent_list):
            current_intent += "Once you find the result, please directly yield a stop action, and give a brief explanation in your answer!"
            # Enhance current intent with previous subtask results if available
            if self.args.subtask and sub_query_idx > 0 and sub_query_answers:
                enhanced_intent = self._enhance_intent_with_previous_results(
                    current_intent, sub_query_answers, sub_query_idx
                )
                self.logger.info(f"[Subtask {sub_query_idx + 1}] Enhanced intent with previous results")
            else:
                enhanced_intent = current_intent
            
            # Reset environment for each sub-query if not the first one
            if sub_query_idx > 0:
                self.logger.info(f"[Sub-query {sub_query_idx + 1}] Resetting environment for new sub-query")
                obs, info = self.env.reset(options={"config_file": config_file})
                current_url = info["page"].url
                state_info: StateInfo = {"observation": obs, "info": info, "current_url": current_url}
                # Clear trajectory and start fresh for new sub-query
                trajectory = [state_info]
                meta_data = {"action_history": [],
                             "page": self.env.page}
                print("CURRENT: ", current_url)
            
            # Process current sub-query
            while True:
                current_url = current_url.lower()

                early_stop_flag, stop_info = early_stop(
                    trajectory, self.args.max_steps, {
                        "parsing_failure": self.args.parsing_failure_th,
                        "repeating_action": self.args.repeating_action_failure_th,
                    }
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    def gen_action(intent, meta):
                        action, meta =  self.agent.next_action_custom(
                            trajectory,
                            intent,
                            meta_data=meta,
                        )
                        return action, meta
                    
                    action, meta_data = gen_action(enhanced_intent, meta_data)
                        
                trajectory.append(action)
                
                action_str = get_action_description(action)
                try:
                    render_helper.render(
                        action, state_info, meta_data, self.args.render_screenshot
                    )
                except Exception as e:
                    self.logger.error(f"Error rendering screenshot: {e}")
                    pass
                meta_data["action_history"].append(action_str)
                meta_data["page"] = self.env.page

                if isinstance(action, list):
                    last_action_type = action[-1]["action_type"]
                else:
                    last_action_type = action["action_type"]
                if last_action_type in [ActionTypes.STOP, 'finished']:
                    self.logger.info(f"[Sub-query {sub_query_idx + 1}] Completed")
                    
                    # Store the subtask answer if using subtask decomposition
                    if self.args.subtask:
                        # Extract the answer from the stop action
                        if isinstance(action, list):
                            answer = action[-1].get('answer', '')
                        else:
                            answer = action.get('answer', '')
                        
                        # Store the subtask intent and answer
                        sub_query_answers.append((enhanced_intent, answer))
                        self.logger.info(f"[Subtask {sub_query_idx + 1}] Answer stored: {answer[:100]}...")
                    
                    break
                try:
                    obs, _, terminated, _, info, current_url = self.env.step(action, observation=obs)
                except Exception as e:
                    self.logger.error(f"Error in step: {e}")
                    terminated = False
                    pass
                # observation, 0.0, done, truncated, info
                print("CURRENT: ", current_url)

                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    self.logger.info(f"[Sub-query {sub_query_idx + 1}] Terminated")
                    
                    # Store the subtask answer if using subtask decomposition
                    if self.args.subtask:
                        sub_query_answers.append((enhanced_intent, "Task terminated without completion"))
                        self.logger.info(f"[Subtask {sub_query_idx + 1}] Terminated without completion")
                    
                    break
                
        # Store trajectory info
        self.trajActions[config_file] = action_list
        
        score, answer_text, ori_answer = self.evaluator(config_file, self.args.result_dir)
        last_action = trajectory[-1]
        pred = last_action.get("answer", "")
        reasoning = last_action.get("reasoning", "")
        self.logger.info(f"[Result] Predicted answer: {pred}\nReasoning: {reasoning}")
        
        self.metrics_dict[config_file] = {
                "config": config_file,
                "success": score,
            }
        self.trajSuccess[config_file] = score

        result = "PASS" if score==1 else "FAIL"
        self.logger.info(f"[Result] ({result}) {config_file}")
        
        render_helper.close()
        
        # End conversation for this task if training data collection is enabled
        if hasattr(self.agent, 'training_collector') and self.agent.training_collector:
            collector = self.agent.training_collector
            if collector and collector.enabled and collector.current_conversation_id:
                # Create conversation summary
                conversation_summary = {
                    "task_id": config_file.split('/')[-1].split('.')[0],
                    "site": site,
                    "sub_domain": '',
                    "success": score,
                    "final_url": current_url,
                    "task_completed": True,
                    "task_description": intent
                }
                
                # End the conversation
                if self.args.save_examples_memory:
                    saved_file = collector.end_conversation(conversation_summary, score)
                    if saved_file:
                        self.logger.info(f"Conversation saved: {saved_file}")
                    else:
                        self.logger.info("Conversation not saved")
                else:
                    self.logger.info("not save_examples_memory")
    
    