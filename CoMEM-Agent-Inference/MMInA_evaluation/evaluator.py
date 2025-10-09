"""Simplified evaluation system for GUI Agent"""
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union
import base64
import re
from PIL import Image

import requests
from beartype import beartype
from playwright.sync_api import CDPSession, Page
from openai import OpenAI

from browser_env import Action, Trajectory, StateInfo
from .helper_functions import (
    clean_answer,
    clean_url,
)
from .helper_functions import encode_image as _encode_image_util


class Evaluator:
    """Base class for evaluation"""
    
    def __init__(self, eval_tag: str = "") -> None:
        self.eval_tag = eval_tag

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page,
        client: CDPSession,
    ) -> float:
        raise NotImplementedError

    @staticmethod
    def get_last_action(trajectory: Trajectory) -> Action:
        """Get the last action from trajectory"""
        if not trajectory or not isinstance(trajectory[-1], dict):
            raise ValueError("The last element of trajectory should be an action")
        return trajectory[-1]

    @staticmethod
    def get_last_state(trajectory: Trajectory) -> StateInfo:
        """Get the last state from trajectory"""
        if len(trajectory) < 2 or not isinstance(trajectory[-2], dict):
            raise ValueError("The second last element of trajectory should be a state")
        return trajectory[-2]


class StringEvaluator(Evaluator):
    """Check whether the answer is correct with exact match, must include, and fuzzy match"""
    
    def __init__(self, vllm_client=None):
        super().__init__()
        self.vllm_client = vllm_client

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page | None = None,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Use helper function for cleaning answers

        last_action = self.get_last_action(trajectory)
        pred = clean_answer(last_action.get("answer", ""))
        
        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    assert isinstance(value, str)
                    ref_answer = clean_answer(value)
                    score = score * (pred == ref_answer)
                case "must_include":
                    url = page.url
                    pred += str(url)
                    assert isinstance(value, list)
                    for must_value in value:
                        must_value = clean_answer(must_value)
                        score = score * (must_value in pred)
                case "fuzzy_match":
                    intent = configs.get("intent", "")
                    assert isinstance(value, list)
                    for reference in value:
                        fuzzy_score = self._llm_fuzzy_match(pred, reference, intent)
                        score = score * fuzzy_score
        return score

    def _llm_fuzzy_match(self, pred: str, reference: str, question: str) -> float:
        """Use vLLM with Qwen2.5-VL-Instruct for binary yes/no matching"""
        try:
            
            # Create the prompt for binary matching
            prompt = f"""You are an evaluator that determines if a predicted answer is correct for a given question.

Question: {question}
Reference Answer: {reference}
Predicted Answer: {pred}

Please evaluate if the predicted answer is correct. Consider:
1. Semantic similarity to the reference answer
2. Key information overlap
3. Factual accuracy
Note: If the predicted answer is like "I'm sorry, I can't answer that question.", you should return "no".

Respond with only "yes" or "no":
- "yes": if the predicted answer is correct or equivalent to the reference answer
- "no": if the predicted answer is incorrect, incomplete, or irrelevant"""

            # Call the model
            response, _, _ = self.vllm_client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful evaluator that provides binary yes/no responses."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=5
            )
            
            # Extract the response
            answer_text = response.content.strip().lower()
            
            # Parse the yes/no response
            if 'yes' in answer_text:
                return 1.0
            elif 'no' in answer_text:
                return 0.0
            else:
                print(f"Could not parse yes/no from response: '{answer_text}'")
                # Fallback: return 0.0 if parsing fails
                return 0.0
                
        except Exception as e:
            print(f"Error in fuzzy matching: {e}")
            # Fallback: return 0.0 if there's an error
            return 0.0
        


class URLExactEvaluator(Evaluator):
    """Check whether the URL is exactly the same as the reference URLs"""
    
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page,
        client: CDPSession | None = None,
    ) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        # Use helper function for cleaning URLs

        pred = clean_url(page.url)
        ref_urls = configs["eval"]["reference_url"].split(" |OR| ")
        ref_urls = [clean_url(url) for url in ref_urls]
        matching_rule = configs["eval"].get("url_note", "EXACT")
        
        if matching_rule == "EXACT":
            if pred in ref_urls:
                return 1.0
            else:
                return 0.0
        elif matching_rule == "GOLD in PRED":
            if any([ref in pred for ref in ref_urls]):
                return 1.0
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown matching rule: {matching_rule}")


class HTMLContentEvaluator(Evaluator):
    """Check whether the contents appear in the page"""
    
    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page,
        client: CDPSession | None = None,
    ) -> float:
        def clean(text: str) -> str:
            text = str(text)
            return text.strip().lower()

        with open(config_file, "r") as f:
            configs = json.load(f)

        targets = configs["eval"]["program_html"]
        score = 1.0
        
        for target in targets:
            target_url: str = target["url"]
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", page.url)
                target_url = eval(func)

            required_contents: str = target["required_contents"]
            locator: str = target["locator"]

            # Navigate to that URL
            if target_url != "last":
                page.goto(target_url)
                time.sleep(2)  # Wait for page to load

            # Get the element content
            if not locator.strip():
                selected_element = page.content()
            elif locator.startswith("document."):
                try:
                    selected_element = page.evaluate(f"() => {locator}")
                    if not selected_element:
                        selected_element = ""
                    selected_element = str(selected_element)
                except Exception:
                    selected_element = ""
            else:
                raise ValueError(f"Unknown locator: {locator}")

            required_contents_or = [
                clean(x) for x in required_contents.split(" |OR| ")
            ]
            selected_element = clean(selected_element)
            score *= any([
                content in selected_element
                for content in required_contents_or
            ])

        return score

class EvaluatorComb:
    """Combination of multiple evaluators"""
    
    def __init__(self, evaluators: List[Evaluator]) -> None:
        self.evaluators = evaluators

    def __call__(
        self,
        trajectory: Trajectory,
        config_file: Path | str,
        page: Page,
        client: CDPSession,
    ) -> float:
        score = 1.0
        for evaluator in self.evaluators:
            cur_score = evaluator(trajectory, config_file, page, client)
            score *= cur_score
        return score


def evaluator_router(config_file: Path | str, vllm_client=None) -> EvaluatorComb:
    """Router to get the evaluator class based on config file"""
    
    with open(config_file, "r") as f:
        configs = json.load(f)
    
    eval_types = configs["eval"]["eval_types"]

    evaluators: List[Evaluator] = []
    
    for eval_type in eval_types:
        match eval_type:
            case "string_match":
                evaluators.append(StringEvaluator(vllm_client))
            case "url_match":
                evaluators.append(URLExactEvaluator())
            case "program_html":
                evaluators.append(HTMLContentEvaluator())
            case _:
                raise ValueError(f"eval_type {eval_type} is not supported")

    return EvaluatorComb(evaluators) 