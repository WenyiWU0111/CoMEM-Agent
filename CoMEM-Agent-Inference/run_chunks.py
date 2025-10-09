"""Simplified script to run the GUI Agent with parallel chunk processing for a single domain"""
import os
import glob
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

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
sys.path.insert(0, 'GUI-Agent/')

class ChunkRunner:
    """Handles parallel execution of test chunks within a single domain"""
    
    def __init__(self, args, chunk_size: int = 500, max_concurrent_chunks: int = 5):
        self.args = args
        self.chunk_size = chunk_size
        self.max_concurrent_chunks = max_concurrent_chunks
        
    def chunk_test_files(self, test_files: List[str]) -> List[List[str]]:
        """Split test files into chunks of specified size"""
        chunks = []
        for i in range(0, len(test_files), self.chunk_size):
            chunk = test_files[i:i + self.chunk_size]
            chunks.append(chunk)
        return chunks
    
    def run_chunk(self, chunk: List[str], chunk_idx: int, logger) -> Dict[str, Any]:
        """Run evaluation for a single chunk of test files"""
        logger.info(f"Starting chunk {chunk_idx + 1} with {len(chunk)} test files")
        # Construct agent
        agent_per_chunk = construct_agent(self.args)
        # Run the appropriate test runner for this chunk
        if self.args.evaluation_type == "mmina":
            test_runner = MMInATestRunner(self.args, agent_per_chunk)
            result = test_runner.run(chunk)
        elif self.args.evaluation_type in ["webvoyager", "expand_memory"]:
            test_runner = WebVoyagerTestRunner(self.args, agent_per_chunk)
            result = test_runner.run(chunk)
        elif self.args.evaluation_type == "mind2web":
            test_runner = Mind2WebTestRunner(self.args, agent_per_chunk)
            result = test_runner.run(chunk)
        else:
            raise ValueError(f"Unsupported evaluation type for chunking: {self.args.evaluation_type}")
        
        logger.info(f"Completed chunk {chunk_idx + 1}")
        return {
            'status': 'completed',
            'chunk_idx': chunk_idx,
            'chunk_size': len(chunk),
            'result': result
        }
    
    def get_test_files_for_domain(self, domain: str, logger) -> List[str]:
        """Get test files for a specific domain and evaluation type"""
        if self.args.evaluation_type == "mmina":
            test_file_list = create_test_file_list_mmina(domain)
            if not self.args.debug:
                test_file_list = get_unfinished(test_file_list, self.args.result_dir, 'mmina')
            return test_file_list
        elif self.args.evaluation_type == "webvoyager":
            test_file_list = create_test_file_list_webvoyager(domain, self.args.test_start_idx, self.args.test_end_idx)
            if not self.args.debug:
                test_file_list = get_unfinished(test_file_list, self.args.result_dir, 'webvoyager_evaluation')
            return test_file_list
        elif self.args.evaluation_type == "mind2web":
            test_file_list = glob.glob(os.path.join(f"Mind2Web/evaluation_data/{domain}", "**", "*.json"), recursive=True)
            if not self.args.debug:
                test_file_list = get_unfinished(test_file_list, self.args.result_dir, 'Mind2Web/evaluation_data')
            return test_file_list
        elif self.args.evaluation_type == "expand_memory":
            test_file_list = create_test_file_list_expand_memory(domain, self.args.test_start_idx, self.args.test_end_idx)
            if not self.args.debug:
                test_file_list = get_unfinished(test_file_list, self.args.result_dir)
            return test_file_list
        else:
            logger.warning(f"Unknown evaluation type: {self.args.evaluation_type}")
            return []
    
    def run_evaluation_with_chunks(self, domain: str, logger):
        """Run evaluation for a domain with parallel chunk processing"""
        
        # Get test files for this domain
        test_files = self.get_test_files_for_domain(domain, logger)
        
        if not test_files:
            logger.warning(f"No test files found for domain {domain}")
            return {
                'status': 'completed',
                'total_chunks': 0,
                'chunk_results': []
            }
        
        # Split test files into chunks
        chunks = self.chunk_test_files(test_files)
        logger.info(f"Split {len(test_files)} test files into {len(chunks)} chunks")
        
        # Run chunks in parallel using ThreadPoolExecutor
        chunk_results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent_chunks) as executor:
            # Submit all chunk tasks
            future_to_chunk = {
                executor.submit(self.run_chunk, chunk, chunk_idx, logger): chunk_idx
                for chunk_idx, chunk in enumerate(chunks)
            }
            
            # Wait for all chunks to complete
            for future in future_to_chunk:
                chunk_idx = future_to_chunk[future]
                result = future.result()
                chunk_results.append(result)
                logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} completed")
        
        # Aggregate results
        completed_chunks = len([r for r in chunk_results if r['status'] == 'completed'])
        failed_chunks = len([r for r in chunk_results if r['status'] == 'failed'])
        
        logger.info(f"Completed: {completed_chunks}/{len(chunks)} chunks successful")
        
        return {
            'status': 'completed',
            'total_chunks': len(chunks),
            'completed_chunks': completed_chunks,
            'failed_chunks': failed_chunks,
            'chunk_results': chunk_results
        }


def main():
    """Main execution function"""
    # Parse arguments with chunk support
    import argparse
    
    # Get standard arguments
    args = config()
    
    # Add chunk-specific arguments
    parser = argparse.ArgumentParser(description='Chunk-based GUI Agent Runner')
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=500,
        help="Number of test files per chunk (default: 500)"
    )
    parser.add_argument(
        "--max-concurrent-chunks", 
        type=int, 
        default=10,
        help="Maximum number of chunks to process concurrently (default: 10)"
    )
    
    # Parse additional arguments
    chunk_args = parser.parse_known_args()[0]
    
    # Merge the arguments
    args.chunk_size = chunk_args.chunk_size
    args.max_concurrent_chunks = chunk_args.max_concurrent_chunks
    args.sleep_after_execution = 1.0
    
    # Domain is assumed to be a single string
    domain = args.domain
    
    print(f"Processing domain: {domain}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Max concurrent chunks: {args.max_concurrent_chunks}")
    
    # Setup logging
    datetime, LOG_FILE_NAME, logger = setup_logging(args)
    set_global_variables(datetime, LOG_FILE_NAME, logger)
    
    # Prepare environment
    prepare(args)
    logger.info(f"Domain: {domain}")
    logger.info(f"Observation context length: {args.max_obs_length}")
    logger.info(f"Use memory: {args.use_memory}")
    
    # Load grounding model using vLLM
    grounding_model = load_grounding_model_vllm(args)
    args.grounding_model = grounding_model
    
    # Create and run chunk runner
    runner = ChunkRunner(
        args, 
        chunk_size=args.chunk_size,
        max_concurrent_chunks=args.max_concurrent_chunks
    )
    
    # try:
    result = runner.run_evaluation_with_chunks(domain, logger)
    
    # Print summary
    print("\n" + "="*60)
    print("CHUNK EVALUATION SUMMARY")
    print("="*60)
    
    if result['status'] == 'completed':
        if 'total_chunks' in result and result['total_chunks'] > 0:
            completed = result.get('completed_chunks', 0)
            total = result.get('total_chunks', 0)
            print(f"✓ {domain}: COMPLETED ({completed}/{total} chunks)")
        else:
            print(f"✓ {domain}: COMPLETED (no chunking)")
    else:
        print(f"✗ {domain}: FAILED")
    
    print("="*60)
    logger.info(f"Evaluation finished. Log file: {LOG_FILE_NAME}")


if __name__ == "__main__":
    main()
