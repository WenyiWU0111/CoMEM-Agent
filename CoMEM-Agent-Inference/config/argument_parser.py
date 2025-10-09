"""Argument parser configuration for the GUI Agent"""
import argparse

def config() -> argparse.Namespace:
    """Configure and parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    # Browser environment arguments
    parser.add_argument(
        "--render", action="store_true", help="Render the browser"
    )
    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--observation_type",
        choices=["accessibility_tree", "html", "image"],
        default="image",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=720)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument(
        "--imgbin_dir",
        type=str,
        default="",
    ) # Not in use

    # Agent configuration
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failure exceeds this threshold, the agent will stop",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeating action exceeds this threshold, the agent will stop",
        type=int,
        default=2,
    )
    parser.add_argument("--domain", type=str, default="shopping")
    parser.add_argument("--hist", action='store_true', default=False)
    parser.add_argument("--hist_fold", type=str, default="./cache/history/")
    parser.add_argument("--hist_num", type=int, default=1)

    parser.add_argument("--task_cnt", type=int, default=0)
    parser.add_argument("--hop_cnt", type=int, default=0)
    
    # API key
    parser.add_argument('--open_router_api_key', type=str, default='', help='OpenRouter API key')
    
    # Language model configuration
    parser.add_argument("--provider", type=str, default="custom")
    parser.add_argument("--model", type=str, default="qwen2.5-vl")
    parser.add_argument("--loaded_tokenizer", default=None)
    parser.add_argument("--loaded_model", default=None)
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--use_memory", type=bool, default=False)
    parser.add_argument("--use_history", type=bool, default=False)
    parser.add_argument("--use_continuous_memory", type=bool, default=False)
    parser.add_argument("--faiss_index_path", type=str, default='')
    parser.add_argument("--similar_num", type=int, default=3)
    parser.add_argument("--bank_size", type=int, default=None)
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=1920,
    )
    parser.add_argument("--add_history_num", type=int, default=5, 
                       help="Whether to add history actions to the prompt")
    parser.add_argument("--save_examples_memory", action='store_true', default=True, 
                       help="Whether to add example memory to the agent")
    parser.add_argument("--instruction_jsons", type=str, nargs='+', default=[], 
                       help="jsons to use for example retrieval")
    
    # Example configuration
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=10000)

    # Logging related
    parser.add_argument("--result_dir", type=str, default="")
    
    # Training data collection configuration
    parser.add_argument("--collect_training_data", action='store_true', default=True,
                        help="Enable collection of training data (prompts and responses)")
    parser.add_argument("--training_data_dir", type=str, default="training_data",
                        help="Directory to save training data files")
    
    # Subtask decomposition configuration
    parser.add_argument("--subtask", action='store_true', default=False, 
                       help="Enable subtask decomposition for complex task breakdown")
    
    # Evaluation configuration
    parser.add_argument("--evaluation_type", type=str, default="mmina", 
                    #    choices=['mmina', 'supergpqa', 'webwalkerqa','expand_memory',  'visualwebarena', 'webarena', 'webvoyager', 'mind2web'],
                       help="Type of evaluation to run")
    parser.add_argument("--render_screenshot", action='store_true', 
                       help="Render screenshots during evaluation")
    
    # WebWalkerQA evaluation configuration
    parser.add_argument("--webwalkerqa_split", type=str, default="silver", 
                       choices=['main', 'silver'])
    
    # Manual Action Instruction
    parser.add_argument("--manual_action", action='store_true', default=False, 
                       help="Enable manual action instruction for complex task breakdown")
    parser.add_argument("--debug", action='store_true', default=False, 
                       help="Enable debug mode")
    
    parser.add_argument("--datetime", type=str, default=None)
    
    args = parser.parse_args()
    args.instruction_path = 'agent/prompts/jsons/p_cot_ground_actree_2s.json'
    if args.use_continuous_memory:
        args.model = 'agent-qformer'
    if args.datetime is None:
        datetime = 'test'
        args.datetime = datetime
    # Set result directory based on evaluation type
    if not args.result_dir:
        args.result_dir = f'results/{args.evaluation_type}/{args.model}/{args.domain}/{args.datetime}'
        
    # Set training data directory
    args.training_data_dir = f"training_data/{args.evaluation_type}/{args.domain}/{args.model}/{args.datetime}"
    
    return args 
