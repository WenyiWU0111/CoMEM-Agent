import sys
sys.path.append('GUI-Agent')
from agent.llm_config import create_direct_vllm_model
from evaluator import LLMEvaluator
import os
import json
import argparse


class LLMEvaluation:
    def __init__(self, html_folder: str, args: argparse.Namespace):
        self.llm_client = create_direct_vllm_model(args, model_name=args.model)
        self.html_folder = html_folder
        self.config_folder = args.config_folder
        self.evaluator = LLMEvaluator(self.llm_client)
        
    def evaluate(self):
        results = []
        all_scores = []
        for config_file in os.listdir(self.config_folder):
            config_path = os.path.join(self.config_folder, config_file)
            with open(config_path, 'r') as f:
                config = json.load(f)
                task_id = config.get('task_id', '')
                html_file = os.path.join(self.html_folder, f"render_{task_id}.html")
            print(f"Evaluating task {task_id} with html file {html_file}")
            if not os.path.exists(html_file):
                continue
            score, answer_text, ori_answer = self.evaluator(config_path, self.html_folder)
            all_scores.append(score)
            results.append({'task_id': task_id, 'score': score, 'answer_text': answer_text, 'ori_answer': ori_answer})
            print(f"Task {task_id} score: {score}")
            with open(os.path.join(self.html_folder, f"llm_evaluation.json"), 'w') as f:
                json.dump(results, f, indent=4)
        print('html_folder: ', self.html_folder)
        print(f"Average score: {sum(all_scores) / len(all_scores)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--html_folder', type=str, default='results/webvoyager/websight/test/baseline')
    parser.add_argument('--model', type=str, default='qwen2.5-vl-32b')
    parser.add_argument('--config_folder', type=str, default='webvoyager_evaluation/data/test')
    args = parser.parse_args()
    llm_evaluation = LLMEvaluation(args.html_folder, args)
    llm_evaluation.evaluate()