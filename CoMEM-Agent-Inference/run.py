"""Main script to run the GUI Agent"""
import os
import glob

from agent import construct_agent
from config.argument_parser import config

from MMInA_evaluation.test_runner import TestRunner as MMInATestRunner
from webvoyager_evaluation.test_runner import TestRunner as WebVoyagerTestRunner
from Mind2Web_evaluation.test_runner import TestRunner as Mind2WebTestRunner

from utils.help_functions import prepare, set_global_variables, get_unfinished, create_test_file_list_mmina, create_test_file_list_webvoyager, create_test_file_list_expand_memory
from utils.logging_setup import setup_logging

# Import the new grounding model loader
from agent.llm_config import load_grounding_model_vllm
import sys
sys.path.insert(0, 'GUI-Agent')

def main():
    """Main execution function"""
    args = config()
    args.sleep_after_execution = 2.5
    
    # Setup logging
    datetime, LOG_FILE_NAME, logger = setup_logging(args)
    set_global_variables(datetime, LOG_FILE_NAME, logger)
    
    # Prepare environment
    prepare(args)
    logger.info(f"Observation context length: {args.max_obs_length}")
    logger.info(f"Use memory: {args.use_memory}")
    
    # Load model and agent
    # model, tokenizer = load_model(args)
    # args.loaded_model = model
    # args.loaded_tokenizer = tokenizer
    
    # Load grounding model using vLLM
    grounding_model = load_grounding_model_vllm(args)
    args.grounding_model = grounding_model
    
    agent = construct_agent(args)
    
    # Run evaluation based on type
    if args.evaluation_type == "mmina":
        # Default MMInA evaluation
        test_file_list = create_test_file_list_mmina(args.domain)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'mmina')
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_mmina(args, agent, test_file_list)
    elif args.evaluation_type == "webvoyager":
        test_file_list = create_test_file_list_webvoyager(args.domain, args.test_start_idx, args.test_end_idx)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'webvoyager_evaluation')
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_webvoyager(args, agent, test_file_list)
    elif args.evaluation_type == "mind2web":
        test_file_list = glob.glob(os.path.join(f"Mind2Web/evaluation_data/{args.domain}", "**", "*.json"), recursive=True)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir, 'Mind2Web/evaluation_data')
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_mind2web(args, agent, test_file_list)
    elif args.evaluation_type == "expand_memory":
        test_file_list = create_test_file_list_expand_memory(args.domain, args.test_start_idx, args.test_end_idx)
        if not args.debug:
            test_file_list = get_unfinished(test_file_list, args.result_dir)
        # print(test_file_list)
        logger.info(f"Total {len(test_file_list)} tasks to process")
        run_tests_webvoyager(args, agent, test_file_list)
    logger.info(f"Test finished. Log file: {LOG_FILE_NAME}")


def run_tests_mmina(args, agent, config_file_list):
    """Run the main test loop"""
    
    test_runner = MMInATestRunner(args, agent)
    test_runner.run(config_file_list)

def run_tests_webvoyager(args, agent, config_file_list):
    """Run the main test loop"""
    
    test_runner = WebVoyagerTestRunner(args, agent)
    test_runner.run(config_file_list)

def run_tests_mind2web(args, agent, config_file_list):
    """Run the main test loop"""

    test_runner = Mind2WebTestRunner(args, agent)
    test_runner.run(config_file_list)


if __name__ == "__main__":
    main()
