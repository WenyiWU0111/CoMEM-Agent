"""Function Call Agent for GUI Agent using direct model calls with ReAct paradigm"""
import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
import base64

from browser_env import Trajectory, Action
from browser_env.actions import ActionTypes
from actions import (
    create_click_action,
    create_selection_action,
    create_type_action,
    create_scroll_action,
    create_wait_action,
    create_stop_action,
    create_none_action,
    create_key_press_action,
    create_goto_url_action,
    parse_action_json,
    # validate_action
)
from .llm_config import create_direct_model, load_tool_llm
from tools.gui_tools import ClickTool, TypeTool, ScrollTool, WaitTool, StopTool, PressKeyTool, PageGotoTool, SelectionTool
from tools.analysis_tools import MapSearchTool, ContentAnalyzerTool
from tools.web_search_tools import WebSearchTool


def resize_image_base64(base64_string: str) -> str:
    """Simple image resize to reduce token count"""
    try:
        from PIL import Image
        import io
        
        # Decode and resize
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        
        # Simple resize to 400x300
        image = image.resize((400, 300), Image.Resampling.LANCZOS)
        
        # Save as JPEG with low quality
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=50)
        output.seek(0)
        
        # Return compressed base64
        return base64.b64encode(output.getvalue()).decode('utf-8')
        
    except:
        return base64_string


class FunctionCallAgent:
    """Custom function call agent for GUI interactions using ReAct paradigm with direct model calls"""
    
    def __init__(self, args: argparse.Namespace, **kwargs):
        """Initialize the function call agent"""
        # Initialize direct model
        self.llm = create_direct_model(args)
        # Initialize tool LLM for tools that need it
        self.tool_llm = load_tool_llm(args)  
        
        # Define functions for the agent
        function_list = self._define_functions()
        # Build dynamic tool specs for prompt
        self.tool_specs = self._build_tool_specs(function_list)
    
        self.args = args
        self.logger = logging.getLogger("logger")
        
        # Initialize function map for tools
        self.function_map = {}
        self._initialize_function_map()
        print('*'*50, 'function_map', '*'*50)
        print(self.function_map)
        print('*'*50, 'function_map', '*'*50)
        
        training_data_dir = getattr(args, 'training_data_dir', 'training_data')
        memory_data_dir = getattr(args, 'memory_data_dir', 'training_data')
        # Initialize training data collector if enabled
        if hasattr(args, 'collect_training_data') and args.collect_training_data:
            from utils.training_data_collector import TrainingDataCollector, get_collector, set_collector
            from utils.llm_wrapper import wrap_llm
            self.training_collector = TrainingDataCollector(
                output_dir=training_data_dir,
                enabled=True
            )
            set_collector(self.training_collector)
            wrapped_llm = wrap_llm(self.llm)
            self.llm = wrapped_llm
        else:
            self.training_collector = None
        
        # Initialize memory system if enabled
        if  hasattr(args, 'use_memory') and args.use_memory:
            from memory.experience_memory import Memory
            # Determine if multimodal memory should be used
            multimodal = True
            # Check if there's a saved index path
            faiss_index_path = getattr(args, 'faiss_index_path', None)
            print(f"Initializing Memory system (multimodal: {multimodal})...")
            self.memory = Memory(training_data_path=memory_data_dir, multimodal=multimodal, faiss_index_path=faiss_index_path, agent=self, bank_size=args.bank_size)
            print("Memory system initialized successfully")
            self.experience_memory = None
            self.experience_texts, self.experience_images, self.file_id_list = None, None, None
        else:
            self.memory = None
        
        # Store analysis results and map search context for next steps
        self.last_analysis_result = None
        self.last_map_search_query = None
        self.last_map_search_result = None
        self.last_page_goto_name = None
        self.last_page_goto_result = None
        self.last_web_search_result = None
        self.last_web_search_screenshots = None

    
    def _define_functions(self) -> List[str]:
        """Define the functions available to the agent"""
        # Use function names instead of tool instances
        functions = [
            'click',
            'type', 
            'press_key',
            'scroll',
            'wait',
            'stop',
            'map_search',
            'content_analyzer',
            # 'goto_homepage',
            # 'goto_url',
            # 'google_web_search'
        ]
        return functions
    
    def _build_tool_specs(self, function_list: List[str]) -> List[Dict[str, Any]]:
        """Build tool specs (name, description, parameters) for prompt from function_list."""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'selection': SelectionTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            # 'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
            }
            
        specs: List[Dict[str, Any]] = []
        for item in function_list:
            cls = name_to_cls.get(str(item))
            tool = None
            if cls is not None:
                tool = cls()
            else:
                print(f"Tool {item} not found")
            if tool is not None:
                parameters = getattr(tool, 'parameters', None)
                if parameters is not None:
                    params_info = {}
                    parameters = parameters.get('properties', {})
                    for param_name, param_info in parameters.items():
                        params_info[param_name] = param_info
                    parameters['properties'] = params_info
                specs.append({
                    'name': getattr(tool, 'name', str(item)),
                    'description': getattr(tool, 'description', 'No description'),
                    'parameters': parameters
                })
    
        return specs
    
    def _initialize_function_map(self):
        """Initialize the function map with tool instances"""
        name_to_cls = {
            # Action tools
            'click': ClickTool,
            'selection': SelectionTool,
            'type': TypeTool,
            'scroll': ScrollTool,
            'wait': WaitTool,
            'stop': StopTool,
            'press_key': PressKeyTool,
            # Analysis tools
            'map_search': MapSearchTool,
            'content_analyzer': ContentAnalyzerTool,
            # 'goto_homepage': GotoHomepageTool,
            # 'goto_url': PageGotoTool,
            # 'google_web_search': WebSearchTool
        }
        
        for name, cls in name_to_cls.items():
            try:
                tool = cls()
                tool.llm = self.tool_llm  # Set the tool LLM
                self.function_map[name] = tool
            except Exception as e:
                print(f"Failed to initialize tool {name}: {e}")
    
    def _get_system_message(self, intent, trajectory) -> str:
        """Get the system message for the agent using ReAct paradigm"""
        tools_section = ""
        lines = []
        for spec in self.tool_specs:
            desc = spec.get('description') or ''
            name = spec.get('name')
            params = spec.get('parameters', {})
            
            # Build parameter description
            param_desc = ""
            if params and 'properties' in params:
                param_list = []
                for param_name, param_info in params['properties'].items():
                    param_type = param_info.get('type', 'string')
                    param_desc_text = param_info.get('description', '')
                    if 'enum' in param_info:
                        enum_values = ', '.join(param_info['enum'])
                        param_list.append(f"`{param_name}` ({param_type}: {enum_values})")
                    else:
                        param_list.append(f"`{param_name}` ({param_type}): {param_desc_text}")
                param_desc = f" - Parameters: {', '.join(param_list)}"
            
            lines.append(f"- **{name}**: {desc}{param_desc}")
        tools_section = "\n".join(lines)
        if self.args.use_memory and self.memory is not None:
            first_image = self._get_first_screenshot(trajectory)
            if self.experience_memory is None:
                print(f'constructing experience memory with similar_num: {self.args.similar_num}')
                self.experience_memory, self.experience_texts, self.experience_images, self.file_id_list = self.memory.construct_experience_memory(intent, self, current_image=first_image, 
                                                                                 dataset=self.args.evaluation_type, domain=self.args.domain, 
                                                                                 similar_num=self.args.similar_num)
        else:
            with open(f"agent/prompts/examples.txt", 'r') as f:
                self.experience_memory = f.read()
                
        with open(f"agent/prompts/system_prompt.txt", 'r') as f:
            system_prompt = f.read()
        system_prompt = system_prompt.format(experience_memory=self.experience_memory, tools_section=tools_section)
        return system_prompt

    def next_action_custom(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: Dict[str, Any],
    ):
        """Generate the next action using function calling with ReAct paradigm"""
        
        self.current_step += 1
        print('*'*50, 'current step: ', self.current_step, '*'*50)
        # Prepare messages for the LLM
        messages, meta_data = self._prepare_messages(trajectory, intent, meta_data)
        
        if self.args.use_continuous_memory and self.memory is not None:
            responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False, 
                                        experience_texts=self.experience_texts, experience_images=self.experience_images,
                                        file_id_list=self.file_id_list)
        else:
            # Call the LLM with function calling
            responses, original_inputs, original_outputs = self.llm.chat(messages=messages, stream=False)
        meta_data['original_inputs'] = original_inputs
        meta_data['original_outputs'] = original_outputs
        meta_data['original_responses'] = responses
        if self.training_collector:
            self.llm.save_conversation(messages, responses.content)
        if not isinstance(responses, list):
            responses = [responses]
        print('*'*50, 'responses', '*'*50)
        print(responses[0].content)
        print('*'*50, 'responses', '*'*50)
        meta_data['response_history'].extend([response.content for response in responses])
                
        # Extract page if available in meta_data or elsewhere; pass explicitly
        page_for_tools = meta_data.get('page')
            
        # Process the response
        action = self._process_response(responses, trajectory, page_for_tools, intent)
        print('*'*50, 'action', '*'*50)
        print(action)
        print('*'*50, 'action', '*'*50)
        
        return (action, meta_data)
    
    def _get_current_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the current screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[-1]
        if isinstance(recent_obs, dict) and 'observation' in recent_obs:
            obs = recent_obs['observation']
            if 'image' in obs:
                return obs['image']
        return None
    
    def _get_first_screenshot(self, trajectory: Trajectory) -> Optional[str]:
        """Extract the first screenshot from the trajectory for multimodal memory retrieval."""
        recent_obs = trajectory[0]
        return recent_obs['observation']['image']
    
    def _prepare_messages(self, trajectory: Trajectory, intent: str, meta_data: Dict[str, Any]) -> List[Dict]:
        """Prepare messages for the LLM with ReAct context"""
        messages = []
        
        # Add system message    
        messages.append({
            'role': 'system',
            'content': self._get_system_message(intent, trajectory)
        })
        
        # Add current intent with ReAct prompt
        current_task = intent
        
        # Add analysis results context if available
        if self.last_analysis_result:
            analysis_summary = self.last_analysis_result
            messages.append({
                'role': 'user',
                'content': analysis_summary
            })
            # Clear the analysis result after using it
            self.last_analysis_result = None
        
        # Add web search results context if available
        if self.last_web_search_result:
            web_search_summary = self.last_web_search_result
            # Create content with text and screenshots if available
            if self.last_web_search_screenshots:                
                content_items = [
                    {"type": "text", "text": f"**Web search results:** {web_search_summary}"}
                ]
                # Add screenshot images and collect paths for cleanup
                screenshot_files_to_delete = []
                for screenshot_path in self.last_web_search_screenshots:
                    if os.path.exists(screenshot_path):
                        # Convert image to base64
                        import base64
                        with open(screenshot_path, 'rb') as img_file:
                            img_data = img_file.read()
                            img_base64 = base64.b64encode(img_data).decode('utf-8')
                            content_items.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        screenshot_files_to_delete.append(screenshot_path)
                
                messages.append({
                    'role': 'user',
                    'content': content_items
                })
                
                # Delete screenshot files after adding them to messages
                for screenshot_path in screenshot_files_to_delete:
                    try:
                        os.remove(screenshot_path)
                    except Exception as e:
                        continue
            else:
                messages.append({
                    'role': 'user',
                    'content': f"**Web search results:** {web_search_summary}"
                })
            # Clear the web search result after using it
            self.last_web_search_result = None
            self.last_web_search_screenshots = None
        
        # Add page goto results context if available
        if self.last_page_goto_result:
            page_goto_summary = f"Successfully navigated to {self.last_page_goto_name} at {self.last_page_goto_result}"
            messages.append({
                'role': 'user',
                'content': f"**Page navigation results:** {page_goto_summary}"
            })
            
            # Clear the page goto result after using it
            self.last_page_goto_result = None
            self.last_page_goto_name = None
        
        # Add recent trajectory information
        if trajectory:
            recent_obs = trajectory[-1]
            if isinstance(recent_obs, dict) and 'observation' in recent_obs:
                obs = recent_obs['observation']
                if 'image' in obs:
                    # Generate a description of the current page using LLM
                    page_description = self._generate_page_description(obs["image"])
                    # Add the current screenshot with generated description
                    messages.append({
                        'role': 'user',
                        'content': [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{obs['image']}"
                                }
                            },
                            {"type": "text", "text": page_description}
                        ]
                    })
        # Add action history
        if meta_data and 'action_history' in meta_data:
            if self.args.use_history:
                action_history = meta_data['action_history'][-5:]  # Last 5 actions
                history_text = ""
                if action_history:
                    history_text += "**Recent Actions:**\n" + "\n".join(
                        f"{i+1}. {action}" for i, action in enumerate(action_history))
                if history_text:
                    history_text += "\n\n **Please carefully check the LOGIC of your previous actions and reasoning process! DO NOT REPEAT previous action sequences! If you observe repeat in your previous actions, the repeated action is WRONG and DOES NOT WORK! DO ADJUST your action choice at this time!**"
                    messages.append({
                        'role': 'user',
                        'content': history_text
                    })
            action_number_left = getattr(self.args, 'max_steps', 15) - len(meta_data['action_history'])
            if action_number_left > 0:
                messages.append({
                    'role': 'user',
                    'content': f"ACTION NUMBER LEFT: You have **{action_number_left} actions left**, You MUST finish the task within the remaining actions! If the left action number is 1, YOU MUST yield the STOP action and provide the answer!"
                })
        messages.append({
            'role': 'user',
            'content': f"""**Current task:** {current_task}

IMPORTANT REMINDERS:
- Please specify the number label of the item you want to interact with, in the description of the action.
- Don't repeat the same action multiple times - try different approaches if something doesn't work
- If your previous action is type, then you must click related pages or scroll pages to find the information you need.
- Pay attention to images, for many questions about the shape, color, location, etc., you can answer according to the images.
- If the current search term yields no results, you must adjust your search term and try again.
- You must provide an answer within the remaining steps.
- Only output one action at a time, do not output multiple actions at once

What would you like to do next?"""})
        
        return messages, meta_data
    
    def _process_response(self, responses: List[Dict], trajectory: Trajectory, page: Optional[Any] = None, intent: Optional[str] = None) -> Action:
        """Process the LLM response and convert to Action"""
        manual_action = getattr(self.args, 'manual_action', False)
        for response in responses:
            if not manual_action:
                response = parse_action_json(response.content)
            try:
                func_name = response['function_call']['name']
                func_args = response['function_call']['arguments']
                
                # Handle different function calls using the actions module
                if func_name == 'click':
                    return create_click_action(
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        description=func_args.get('description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name in ['selection', 'select']:
                    return create_selection_action(
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        description=func_args.get('description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name in ['type', 'search']:
                    return create_type_action(
                        text=func_args.get('text', ''),
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        field_description=func_args.get('field_description', ''),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'scroll':
                    return create_scroll_action(
                        direction=func_args.get('direction', 'down'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'wait':
                    return create_wait_action(
                        seconds=2.0,  # Default as requested
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'press_key':
                    return create_key_press_action(
                        key_comb=func_args.get('key', 'enter'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'stop':
                    answer = func_args.get('answer', 'Task completed')
                    self.logger.info(f"Agent answer: {answer}")
                    return create_stop_action(
                        answer=func_args.get('answer', 'Task completed'),
                        reasoning=func_args.get('reasoning', '')
                    )
                elif func_name == 'content_analyzer':
                    # Execute content analyzer and store results for next step
                    # Get the tool from function_map
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Add trajectory context, page (if available), and LLM to kwargs
                        kwargs = {'page': page}
                        tool.llm = self.tool_llm
                        result = tool.call(json.dumps(func_args), **kwargs)
                        # self.logger.info(f"Content analyzer result: {result}")
                        
                        # Store the analysis result for next step context
                        # ContentAnalyzerTool returns JSON string, so store it directly
                        self.last_analysis_result = result
                        
                        # Return a wait action to allow the agent to process the analysis result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                        )
                    else:
                        self.logger.error(f"Tool {func_name} not found in function_map")
                        return create_none_action()
                    
                elif func_name == 'map_search':
                    # Execute map search tool which now returns a Google Maps URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string; try to extract
                        url = result.strip()
                        # Store context
                        self.last_map_search_query = func_args.get('query', '')
                        self.last_map_search_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'goto_url':
                    # Execute page goto tool which returns a target URL
                    tool = self.function_map.get(func_name)
                    if tool:
                        tool.llm = self.tool_llm
                        result = tool.call(json.dumps(func_args))
                        # Expect result to be a URL string
                        url = result.strip()
                        # Store context
                        self.last_page_goto_name = func_args.get('page_name', '')
                        self.last_page_goto_result = result
                        # If we got a URL, emit a goto action so the env updates the page
                        return create_goto_url_action(url)
                        
                elif func_name == 'google_web_search':
                    return create_type_action(
                        text=func_args.get('text', ''),
                        element_id=func_args.get('element_id', ''),
                        coords=func_args.get('coords', ''),
                        field_description=func_args.get('field_description', 'search input field'),
                        reasoning=func_args.get('reasoning', '')
                    )
                    # Execute web search tool and store results for next step
                    tool = self.function_map.get(func_name)
                    if tool:
                        # Set the LLM for the web search tool
                        tool.set_llm(self.tool_llm)
                        
                        # Handle async call
                        import asyncio
                        import concurrent.futures
                        
                        # Check if event loop is already running
                        try:
                            # Try to get running loop
                            asyncio.get_running_loop()
                            # Loop is running, use ThreadPoolExecutor
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                                result = future.result(timeout=120)  # 2 minute timeout
                        except RuntimeError:
                            # No loop running, can use asyncio.run directly
                            result = asyncio.run(tool.call(json.dumps(func_args)))
                        # self.logger.info(f"Web search result: {result}")
                        
                        # Extract screenshot information from result
                        screenshot_paths = []
                        if result and "[Screenshot available:" in result:
                            import re
                            screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                            screenshot_paths = screenshot_matches
                            self.logger.info(f"Found screenshots: {screenshot_paths}")
                        
                        # Store the web search result and screenshots for next step context
                        self.last_web_search_result = result
                        self.last_web_search_screenshots = screenshot_paths
                        
                        # Return a wait action to allow the agent to process the search result
                        return create_wait_action(
                            seconds=1.0,
                            reasoning=f"Web search completed. Search results will be available for the next step."
                        )

            except Exception as e:
                print('*'*50, 'no function call', '*'*50)
        # If no function call, try to parse natural language responses
        for response in responses:
            content = response.content
            print(f"Parsing natural language content: {content}")
            
            # If pattern matching fails, use LLM to parse the content
            action = self._parse_natural_language_with_llm(content, page)
            if action and action.get('action_type') != ActionTypes.NONE:
                print(f"Successfully parsed natural language action with LLM: {action}")
                return action
        
        # If no function call and no natural language action found, return none action
        action = create_none_action()

        return action
    

    def _parse_natural_language_with_llm(self, content: str, page: Optional[Any] = None, pure_text=False) -> Action:
        """Use LLM to parse natural language content and extract action information"""
        try:
            # Create a prompt for the LLM to parse the content
            system_prompt = """You are an expert at parsing natural language responses and converting them into structured actions for a GUI automation agent.

Available actions:
- click: Click on elements by describing what you want to click
- selection: Select an option from a dropdown menu by describing what you want to select
- type: Type text into input fields by describing the field
- press_key: Press specific keys (enter, delete, space, etc.)
- scroll: Scroll the page in different directions (up, down, left, right)
- wait: Wait for a specified number of seconds
- stop: Stop the task and provide final answer
- map_search: Navigate to Google Maps for geographical searches
- content_analyzer: Analyze page content and images (results will be available for next step)

Parse the given content and return a key-value list with the following structure:

action_type: click|selection|type|press_key|scroll|wait|stop|map_search|content_analyzer
element_id: id of the element to interact with (for click, selection and type action)
coords: coordinates of the element to interact with (for click, selection and type action), in the format of "<point>x1 y1</point>", and it should be valid with two numbers, without any other text!
description: description of what to click or select (for click and selection action)
text: text to type (for type action)
field_description: description of the field (for type action)
key: key to press (for press_key action)
direction: scroll direction (for scroll action)
seconds: number of seconds (for wait action)
answer: final answer (for stop action)
query: query to search (for map_search action)
reasoning: why this action is needed

EXAMPLES:

For a click action:

action_type: click
element_id: x
coords: <point>x1 y1</point>
description: the search button
reasoning: Need to click the search button to submit the query

For a selection action:

action_type: selection
element_id: x
coords: <point>x1 y1</point>
description: the price option of the dropdown menu want to select from
reasoning: Need to select the price option of the dropdown menu to select the option

For a type action:

action_type: type
element_id: x
coords: <point>x1 y1</point>
text: Sydney Opera House
field_description: the search input field
reasoning: Need to type the search query into the search field

For a press_key action:

action_type: press_key
key: enter
reasoning: Need to press the enter key to submit the query

For a scroll action:

action_type: scroll
direction: down
reasoning: Need to scroll down to load more content


For a wait action:

action_type: wait
seconds: 2.0
reasoning: Need to wait for 2 seconds to load the page


For a stop action (task completion):

action_type: stop
answer: Yes, there is a Ferris wheel in the center of Shanghai. It is the Sky Ring Ferris Wheel in Joy City Shanghai.
reasoning: The information confirms the presence of the Sky Ring Ferris Wheel in the center of Shanghai

For a map_search action:

action_type: map_search
query: Sydney Opera House
reasoning: Need to search for the Sydney Opera House on Google Maps

For a content_analyzer action:

action_type: content_analyzer
reasoning: Need to analyze the page content and images


IMPORTANT: 
1. If the content indicates that a task is complete or provides a final answer, use action_type "stop" with the answer.
2. Ensure all key-value pairs are on separate lines.
3. Values should not contain extra quotes.

REMEMBER: Output ONLY valid key-value pairs, nothing else."""

            user_prompt = f"Parse this content and extract the action:\n\n{content.split('assistant')[0]}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Call the LLM
            result = ''
            while result == '':
                response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
                result = response.content
                print(f"LLM response: {result}")
                if result != '':
                    break
            
            # Extract individual fields
            action_data = {}
            key_list = ['action_type', 'element_id', 'reasoning', 'coords',
                        'description', 
                        'text', 'field_description', 
                        'page_name', 'url'
                        'query',
                        'key', 'direction', 'seconds', 'answer']
            for line in result.strip().splitlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    if key.strip() in key_list:
                        action_data[key.strip()] = value.strip()
            
            # Check if we found any action data
            if not action_data or 'action_type' not in action_data:
                try:
                    action_data = json.loads(result)
                    action_data['action_type'] = action_data['name']
                    for key, value in action_data['arguments'].items():
                        action_data[key] = value
                except:
                    print("No valid action data found in LLM response")
                    return create_none_action()
            
            # print(f"Extracted action data: {action_data}")
            
            # Convert to Action based on action_type
            action_type = action_data.get('action_type', '').lower()
            reasoning = action_data.get('reasoning', '')
            
            if pure_text:
                return {'name': action_data['action_type'], 'arguments': {'reasoning': action_data['reasoning'],
                                                                          'description': action_data.get('description', '')}}
            
            if action_data['action_type'] == 'goto_url' and 'url' not in action_data:
                page_name = action_data.get('page_name', '')
                page_urls = {
                        'wiki': 'https://www.wikipedia.org/',
                        'ticket': 'https://www.trip.com/flights/',
                        'car': 'https://sg.trip.com/carhire/?channelid=14409&locale=en-SG&curr=USD',
                        'flight': 'https://www.momondo.com/',
                        'hotel': 'https://sg.trip.com/hotels/?locale=en-SG&curr=USD',
                        'shopping': 'http://ec2-3-146-212-252.us-east-2.compute.amazonaws.com:7770/',
                        'event': 'https://www.eventbrite.com/',
                        'map': 'https://www.google.com/maps',
                        'youtube': 'https://www.youtube.com/',
                        'food': 'https://www.timeout.com/',
                        'travel': 'https://www.nomadicmatt.com/',
                        'dollars': 'https://www.xe.com/',
                        'twitter': 'https://twitter.com/home',
                    }
                if 'car' in page_name:
                    page_name = 'car'
                elif 'wiki' in page_name:
                    page_name = 'wiki'
                elif 'ticket' in page_name:
                    page_name = 'ticket'
                elif 'flight' in page_name:
                    page_name = 'flight'
                elif 'hotel' in page_name:
                    page_name = 'hotel'
                elif 'shopping' in page_name:
                    page_name = 'shopping'
                elif 'event' in page_name:
                    page_name = 'event'
                elif 'map' in page_name:
                    page_name = 'map'
                elif 'youtube' in page_name:
                    page_name = 'youtube'
                elif 'food' in page_name:
                    page_name = 'food'
                elif 'travel' in page_name:
                    page_name = 'travel'
                elif 'dollars' in page_name:
                    page_name = 'dollars'
                elif 'twitter' in page_name:
                    page_name = 'twitter'
                try:
                    action_data['url'] = page_urls[page_name]
                except:
                    action_data['url'] = ''
            
            if action_type == 'click':
                return create_click_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['selection', 'select']:
                return create_selection_action(
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    description=action_data.get('description', ''),
                    reasoning=reasoning
                )
            elif action_type in ['type', 'search']:
                return create_type_action(
                    text=action_data.get('text', ''),
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    field_description=action_data.get('field_description', ''),
                    reasoning=reasoning
                )
            elif action_type == 'press_key':
                return create_key_press_action(
                    key_comb=action_data.get('key', 'enter'),
                    reasoning=reasoning
                )
            elif action_type == 'scroll':
                return create_scroll_action(
                    direction=action_data.get('direction', 'down'),
                    reasoning=reasoning
                )
            elif action_type == 'wait':
                return create_wait_action(
                    seconds=float(action_data.get('seconds', 2.0)),
                    reasoning=reasoning
                )
            elif action_type == 'stop':
                return create_stop_action(
                    answer=action_data.get('answer', 'Task completed'),
                    reasoning=reasoning
                )
            elif action_type == 'map_search':
                tool = self.function_map.get(action_type)
                if tool:
                    func_args = {
                        'query': action_data.get('query', action_data.get('reasoning', '')),
                        'reasoning': action_data.get('reasoning', '')
                    }
                    result = tool.call(json.dumps(func_args))
                    # Expect result to be a URL string; try to extract
                    url = result.strip()
                    # Store context
                    self.last_map_search_query = func_args.get('query', '')
                    self.last_map_search_result = result
                    # If we got a URL, emit a goto action so the env updates the page
                    return create_goto_url_action(url)
            elif action_type == 'goto_url':
                return create_goto_url_action(
                    url=action_data.get('url', '')
                )
            elif action_type == 'google_web_search':
                return create_type_action(
                    text=action_data.get('text', ''),
                    element_id=action_data.get('element_id', ''),
                    coords=action_data.get('coords', ''),
                    field_description=action_data.get('field_description', 'search input field'),
                    reasoning=reasoning
                )
                # Set the LLM for the web search tool
                tool = self.function_map.get(action_type)
                if tool:
                    tool.set_llm(self.tool_llm)
                    # Handle async call
                    import asyncio
                    import concurrent.futures
                    
                    # Check if event loop is already running
                    try:
                        # Try to get running loop
                        asyncio.get_running_loop()
                        # Loop is running, use ThreadPoolExecutor
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(asyncio.run, tool.call(json.dumps(func_args)))
                            result = future.result(timeout=120)  # 2 minute timeout
                    except RuntimeError:
                        # No loop running, can use asyncio.run directly
                        result = asyncio.run(tool.call(json.dumps(func_args)))
                    # self.logger.info(f"Web search result: {result}")
                    
                    # Extract screenshot information from result
                    screenshot_paths = []
                    if result and "[Screenshot available:" in result:
                        import re
                        screenshot_matches = re.findall(r'\[Screenshot available: ([^\]]+)\]', result)
                        screenshot_paths = screenshot_matches
                        self.logger.info(f"Found screenshots: {screenshot_paths}")
                    
                    # Store the web search result and screenshots for next step context
                    self.last_web_search_result = result
                    self.last_web_search_screenshots = screenshot_paths
                    
                    # Return a wait action to allow the agent to process the search result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Web search completed. Search results will be available for the next step."
                    )
            elif action_type == 'content_analyzer':
                tool = self.function_map.get(action_type)
                func_args = {
                    'query': action_data.get('query', ''),
                    'reasoning': action_data.get('reasoning', '')
                }
                if tool:
                    # Add trajectory context, page (if available), and LLM to kwargs
                    kwargs = {'page': page}
                    tool.llm = self.tool_llm
                    result = tool.call(json.dumps(func_args), **kwargs)
                    # self.logger.info(f"Content analyzer result: {result}")
                    
                    # Store the analysis result for next step context
                    # ContentAnalyzerTool returns JSON string, so store it directly
                    self.last_analysis_result = result
                    
                    # Return a wait action to allow the agent to process the analysis result
                    return create_wait_action(
                        seconds=1.0,
                        reasoning=f"Content analysis completed. Analysis results will be available for the next step."
                    )
            else:
                return create_none_action()
            
        except Exception as e:
            print(f"Error in LLM parsing: {e}")
            return create_none_action()

    
    def _generate_page_description(self, image_base64: str) -> str:
        """Generate a description of the current page using the LLM"""
        try:
            # Create a prompt for the LLM to describe the page
            messages = [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant that analyzes web page screenshots and provides clear, concise descriptions of what you see. Focus on the main content, interactive elements, and overall purpose of the page.'
                },
                {
                    'role': 'user',
                    'content': [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Please describe this web page screenshot. Include the main content, any visible buttons, forms, or interactive elements, and the overall purpose of the page."
                        }
                    ]
                }
            ]
            
            # Get LLM response
            response, _, _ = self.tool_llm.chat(messages=messages, stream=False)
            if hasattr(response, 'content'):
                description = response.content
            else:
                description = str(response)
            
            description = description.replace("\"text\": \"{'role': 'assistant', 'content': '", "")
            description = description.replace("'}\"", "")
            description = description[:2000]
            return description if description else "Current page state - analyze this and decide what to do next"
            
        except Exception as e:
            self.logger.warning(f"Error generating page description: {e}")
            return "Current page state - analyze this and decide what to do next"
    
    def reset(self, test_config_file: str) -> None:
        """Reset the agent for a new task"""
        self.logger.info(f"Resetting agent for config file: {test_config_file}")
        # Clear any internal state if needed
        pass


def construct_agent(args: argparse.Namespace) -> FunctionCallAgent:
    """Construct a function call agent"""
    agent = FunctionCallAgent(args)
    return agent 
