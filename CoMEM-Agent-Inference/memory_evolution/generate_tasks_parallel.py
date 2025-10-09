#!/usr/bin/env python3
"""
Parallel Task Generation Script for Automated Agent Trajectory Rollout System

This script generates tasks for URLs by processing all categories simultaneously:
1. Simplifying URLs to parent pages (e.g., https://www.bbc.com/news/topics/c4gmdg9ne38t -> https://www.bbc.com/news)
2. Using VLM with screenshots and page content from PageParserTool
3. Generating reliable, solvable tasks
4. Processing multiple categories in parallel for better efficiency
"""

import json
import os
import sys
import time
import requests
import argparse
import random
import tracemalloc
from urllib.parse import urlparse, urljoin
from typing import Dict, List, Set, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime
import asyncio

# Add the parent directory to the path to import modules
sys.path.insert(0, 'GUI-Agent')

from agent.llm_config import load_tool_llm
from config.argument_parser import config
from tools.analysis_tools import PageParserTool
from playwright.async_api import async_playwright
from crawl4ai import AsyncWebCrawler

# Constants
TASK_DICTIONARY_PATH = "categories_dict.json"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('memory_evolution/task_generation_parallel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_random_headers() -> Dict[str, str]:
    """Generate random headers for web requests to avoid being blocked"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    return headers

class ParallelTaskGenerator:
    """Parallel task generator for URLs with VLM and page content analysis"""
    
    def __init__(self, selected_categories: List[str] = None, max_concurrent_categories: int = 3):
        """Initialize the parallel task generator"""
        self.config = config()
        self.llm = load_tool_llm(self.config)
        self.page_parser = PageParserTool()
        self.selected_categories = selected_categories
        
        # Semaphore to limit concurrent categories
        self.semaphore = asyncio.Semaphore(max_concurrent_categories)
        
        # Page access statistics (shared across all categories)
        self.successful_accesses = 0
        self.blocked_pages = 0
        self.failed_pages = 0
        self.stats_lock = asyncio.Lock()
        
        # Load task dictionary for examples
        with open(TASK_DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            self.task_dictionary = json.load(f)
        
        # Create output directory
        self.output_dir = Path("memory_evolution/generated_tasks_V4_new")
        self.output_dir.mkdir(exist_ok=True)
    
    def get_examples_for_category(self, category: str, url: str) -> List[str]:
        """
        Get relevant examples from the task dictionary for a given category and URL
        
        Args:
            category: Category name
            url: Target URL
            
        Returns:
            List of example task descriptions
        """
        examples = []
        
        # Try to find examples in the task dictionary
        if category in self.task_dictionary:
            category_data = self.task_dictionary[category]
            
            # Find examples that match the domain or are in the same category
            for item in category_data:
                sample_idxs = random.sample(range(len(item)), min(5, len(item)))
                for idx in sample_idxs:
                    item_url, tasks = list(item.items())[idx]
                    sample_task = random.sample(tasks, 1)[0]
                    examples.append((item_url, sample_task))
        
        # If no domain-specific examples found, use general category examples
        if not examples and category in self.task_dictionary:
            category_data = self.task_dictionary[category]
            all_tasks = []
            
            for item in category_data:
                for tasks in item.values():
                    all_tasks.extend(tasks)
                    break
            
            # Sample up to 5 examples from the category
            if all_tasks:
                examples = random.sample(all_tasks, min(5, len(all_tasks)))
        return examples[:5]  # Return max 5 examples
    
    def simplify_url(self, url: str) -> str:
        """
        Simplify URL to parent page (keep only domain + first path level)
        
        Args:
            url: Original URL
            
        Returns:
            Simplified URL
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.strip('/').split('/')
            
            # Keep only the first path level if it exists
            if path_parts and path_parts[0]:
                simplified_path = f"/{path_parts[0]}"
            else:
                simplified_path = ""
                
            simplified_url = f"{parsed.scheme}://{parsed.netloc}{simplified_path}"
            return simplified_url
            
        except Exception as e:
            logger.error(f"Error simplifying URL {url}: {e}")
            return url
    
    def get_unique_parent_urls(self, categories_dict: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """
        Extract unique parent URLs from categories dictionary
        
        Args:
            categories_dict: Dictionary with categories and URLs
            
        Returns:
            Dictionary with categories and sets of unique parent URLs
        """
        unique_urls = {}
        
        for category, urls in categories_dict.items():
            # Skip categories not in selected_categories if specified
            if self.selected_categories and category not in self.selected_categories:
                continue
                
            parent_urls = set()
            for url in urls:
                if url.strip():  # Skip empty URLs
                    parent_url = self.simplify_url(url)
                    parent_urls.add(parent_url)
            
            unique_urls[category] = parent_urls
            logger.info(f"Category '{category}': {len(urls)} original URLs -> {len(parent_urls)} unique parent URLs")
        
        return unique_urls
    
    async def get_page_content(self, url: str, screenshot: str) -> Tuple[Optional[str], bool]:
        """
        Get page content using PageParserTool and check for blocking
        
        Args:
            url: URL to parse
            screenshot: Base64 screenshot (not used in this implementation)
            
        Returns:
            Tuple of (content, is_blocked)
        """
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)
            return result.markdown, False
            
        except Exception as e:
            async with self.stats_lock:
                self.failed_pages += 1
            logger.error(f"Error getting page content for {url}: {e}")
            return None, True
    
    async def get_page_screenshot(self, page, url: str) -> Optional[str]:
        """
        Get page screenshot using Playwright
        
        Args:
            page: Playwright page instance
            url: URL to screenshot
            
        Returns:
            Base64 encoded screenshot or None if failed
        """
        if not page:
            logger.error(f"Browser page not initialized - cannot take screenshot for {url}")
            return None
            
        try:
            # Navigate to the page
            await page.goto(url, wait_until='networkidle', timeout=30000)
            
            # Take screenshot
            screenshot_bytes = await page.screenshot()
            
            # Convert to base64
            import base64
            screenshot_base64 = base64.b64encode(screenshot_bytes).decode('utf-8')
            
            return screenshot_base64
            
        except Exception as e:
            logger.error(f"Error getting screenshot for {url}: {e}")
            return None
    
    def generate_info_extraction_prompt(self, url: str, category: str, page_content: str) -> str:
        """
        Generate prompt for information extraction from page
        
        Args:
            url: URL to analyze
            category: Category of the URL
            page_content: Content of the page
            
        Returns:
            Prompt for information extraction
        """
        prompt = f"""You are an expert web page analyzer. Analyze the given website and screenshot and extract all useful information for task generation.

URL: {url}
Category: {category}

Page Content:
{page_content[:5000] if page_content else "No content available"}

Based on the URL, category, and page content above, provide a comprehensive summary of:
1. Website type and purpose - its primary intended use cases, problems that can be solved on this site
2. Navigation structure and key sections
3. Interactive elements (buttons, forms, links, etc.)
4. Content types present (articles, product listings, archives, etc.)
5. Anything unusual or unique to this site
6. IGNORE any cookie notices, privacy popups, log in buttons, site errors, ads, or other elements not specific to the site's function.

Focus on information that would be useful for generating problems for web automation to solve. Think about what human users might want to achieve on this website.

Provide a clear, structured summary that captures these essential aspects of the website for task generation purposes."""

        return prompt
    
    def generate_task_prompt(self, url: str, category: str, info_summary: str) -> str:
        """
        Generate prompt for task generation based on extracted information
        
        Args:
            url: URL to generate task for
            category: Category of the URL
            info_summary: Extracted information summary
            
        Returns:
            Prompt for task generation
        """
        # Get relevant examples from the task dictionary
        examples = self.get_examples_for_category(category, url)
        
        # Create examples section
        examples_section = ""
        if examples:
            examples_section = f"""
Here are real examples from the {category} category:

"""
            for i, (example_url, example_task) in enumerate(examples, 1):
                # Clean up the example text (remove the "Only use..." part)
                clean_example_task = example_task.split("Only use")[0].strip()
                examples_section += f"{i}. <WEBSITE_URL: {example_url}> <TASK: {clean_example_task}>\n"
        
        prompt = f"""You are an expert generator of task problems for web automation. Based on the website analysis and screenshot, generate 10 diverse task problems that a web agent could solve on this website.

URL: {url}
Category: {category}

IMPORTANT: Generate direct, actionable problems with a solution. Task problems should be specific and achievable. 
You MUST NOT include overly specific tasks like 'Click on...' or 'Type...'.

Generate exactly 10 diverse, direct instruction task problems that:
1. Have a clear, specific objective; a problem to solve
2. Are achievable within the website's scope
3. Test diverse skills like navigation, information gathering, information synthesis in order to solve problems
4. Require multiple steps to solve, not a single action
5. Have measurable, verifiable success criteria; avoid vague, unverifiable tasks like 'Read a paragraph' or 'Explore a page'. Instead, focus on a clear, deliberate end goal.
6. IMPORTANT: Do NOT write tasks that contain overly specific instructions like 'Click on X...' or 'Type X...'. These are not problems to be solved. The task problem should not direct the agent on what to do, but rather what to achieve. You MUST NOT include wording such as "Click on ...", "or "Type ...".
7. Are distinct, not related to each other, and do not require knowledge or completion of previous task problems.

For each task problem, provide:
- Task problem description (direct instruction with specific requirements)
- Expected outcome (what should be accomplished, what condition needs to be checked to indicate success)
- Difficulty level (easy/medium/hard)

{examples_section}

Format your response as exactly 10 tasks, one per line, with this structure:
1. [Task Description] | [Expected Outcome] | [Difficulty]
2. [Task Description] | [Expected Outcome] | [Difficulty]
3. [Task Description] | [Expected Outcome] | [Difficulty]

Example format:
1. Find a news article about climate change published in the last week | Should locate and display a recent climate change article | Easy
2. Search for iPhone 12 Pro with price below $800 | Should have navigated to a page for iPhone 12 Pro models under $800 | Medium
3. Book a hotel in New York for 2 people for next weekend | Should find and select a specific hotel for 2 people | Medium
4. Find the top 3 customer reviews for Nike running shoes | Should locate and display a page showing the top 3 reviews | Easy
5. Compare prices of Samsung Galaxy phones between $500-700 | Should find a page comparing multiple Samsung phones in that price range | Medium
6. Find the paper with the most citations under the Computer Science category | Should identify and display the page of an AI paper | Hard

Respond only with the 10 tasks in the specified format, no additional text.

Before you finalize the task, THINK: Have I mistakenly included overly specific instructions like 'Click on...' or 'Type...' or 'Scroll to...' in the task description? If so, REWRITE the task to focus on a problem or goal to be solved, not the specific actions to take.

Website Analysis: {info_summary}"""

        return prompt
    
    def parse_task_response(self, response: str, url: str, category: str) -> List[Dict]:
        """
        Parse task response from line format to structured format
        
        Args:
            response: Raw response from VLM
            url: Source URL
            category: Category
            
        Returns:
            List of parsed tasks
        """
        tasks = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            try:
                # Remove line number and parse
                parts = line.split('|')
                if len(parts) >= 3:
                    task_desc = parts[0].split('.', 1)[1].strip() if '.' in parts[0] else parts[0].strip()
                    expected_outcome = parts[1].strip()
                    difficulty = parts[2].strip()
                    
                    task = {
                        "task_description": task_desc,
                        "expected_outcome": expected_outcome,
                        "difficulty": difficulty,
                        "category": category,
                        "url": url,
                        "source_url": url
                    }
                    tasks.append(task)
                    
            except Exception as e:
                logger.warning(f"Failed to parse task line: {line}, error: {e}")
                continue
        
        return tasks
    
    async def generate_tasks_for_url(self, page, url: str, category: str) -> List[Dict]:
        """
        Generate tasks for a specific URL using two-step VLM process
        
        Args:
            page: Playwright page instance
            url: URL to generate tasks for
            category: Category of the URL
            
        Returns:
            List of generated tasks
        """
        try:
            logger.info(f"Generating tasks for {url} (category: {category})")
            
            # Step 1: Get screenshot
            screenshot = await self.get_page_screenshot(page, url)
            if screenshot is None:
                logger.warning(f"Skipping {url} due to screenshot failure")
                return []
            
            # Step 2: Get page content and check for blocking
            page_content, is_blocked = await self.get_page_content(url, screenshot)
            
            if is_blocked:
                logger.warning(f"Skipping {url} due to blocking or access issues")
                return []
            
            # Step 3: Extract information from page
            info_prompt = self.generate_info_extraction_prompt(url, category, page_content)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": info_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
            ]}]
            info_response = self.llm.chat(messages).content
            
            logger.info(f"Extracted information for {url}")
            
            # Step 4: Generate tasks based on extracted information
            task_prompt = self.generate_task_prompt(url, category, info_response)
            messages = [{"role": "user", "content": [
                {"type": "text", "text": task_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot}"}}
            ]}]
            task_response = self.llm.chat(messages).content
            
            # Step 5: Parse tasks
            tasks = self.parse_task_response(task_response, url, category)
            
            logger.info(f"Generated {len(tasks)} tasks for {url}")
            async with self.stats_lock:
                self.successful_accesses += 1
            return tasks
                
        except Exception as e:
            logger.error(f"Error generating tasks for {url}: {e}")
            async with self.stats_lock:
                self.failed_pages += 1
            return []
    
    async def generate_tasks_for_category(self, category: str, urls: Set[str], max_urls: int = 10):
        """
        Generate tasks for all URLs in a category
        
        Args:
            category: Category name
            urls: Set of URLs in the category
            max_urls: Maximum number of URLs to process
            
        Returns:
            List of all generated tasks for the category
        """
        logger.info(f"Processing category '{category}' with {len(urls)} URLs")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Load existing tasks if file exists
        all_tasks = []
        processed_count = 0
        seen_urls = set()
        filename = f"tasks_{category}.json"
        output_path = os.path.join(self.output_dir, filename)
        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                all_tasks = json.load(f)
                seen_urls = set(task['url'] for task in all_tasks)
        logger.info(f"Found {len(seen_urls)} existing URLs for category {category}")

        # Initialize browser for this category
        playwright = None
        browser = None
        page = None
        
        try:
            logger.info(f"Initializing browser for category: {category}")
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=True)
            page = await browser.new_page(viewport={'width': 1280, 'height': 720})
            
            for url in list(urls)[:max_urls]:
                for loop_count in range(5):
                    
                    tasks = await self.generate_tasks_for_url(page, url, category)
                    all_tasks.extend(tasks)
                    if len(tasks) > 0:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(all_tasks, f, indent=2, ensure_ascii=False)
                        logger.info(f"Saved {len(tasks)} tasks for category '{category}' to {output_path} in {loop_count}/5 loops")
                        seen_urls.add(url)
                        break

                processed_count += 1
                
                # Add delay to avoid overwhelming the server
                await asyncio.sleep(2)
                
                if processed_count >= max_urls:
                    break
            
        finally:
            # Cleanup browser
            try:
                if browser:
                    await browser.close()
                if playwright:
                    await playwright.stop()
                logger.info(f"Cleaned up browser for category: {category}")
            except Exception as e:
                logger.error(f"Error cleaning up browser for category {category}: {e}")
        
        logger.info(f"Generated {len(all_tasks)} total tasks for category '{category}'")
        return all_tasks
    
    async def process_category_with_semaphore(self, category: str, urls: Set[str], max_urls: int):
        """
        Process a single category with semaphore control
        
        Args:
            category: Category name
            urls: Set of URLs in the category
            max_urls: Maximum number of URLs to process
        """
        async with self.semaphore:  # Limit concurrent categories
            logger.info(f"Starting processing for category: {category}")
            await self.generate_tasks_for_category(category, urls, max_urls)
            logger.info(f"Completed processing for category: {category}")
    
    def print_statistics(self):
        """Print page access statistics"""
        total_attempted = self.successful_accesses + self.blocked_pages + self.failed_pages
        
        logger.info("=" * 50)
        logger.info("PAGE ACCESS STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total URLs attempted: {total_attempted}")
        logger.info(f"Successfully accessed: {self.successful_accesses}")
        logger.info(f"Blocked pages: {self.blocked_pages}")
        logger.info(f"Failed pages: {self.failed_pages}")
        
        if total_attempted > 0:
            success_rate = (self.successful_accesses / total_attempted) * 100
            logger.info(f"Success rate: {success_rate:.1f}%")
        
        logger.info("=" * 50)
    
    async def run(self, categories_dict: Dict[str, List[str]], max_urls_per_category: int = 5):
        """
        Main execution method - processes all categories in parallel
        
        Args:
            categories_dict: Dictionary with categories and URLs
            max_urls_per_category: Maximum URLs to process per category
        """
        logger.info("Starting parallel task generation process")
        
        # Get unique parent URLs for all categories
        unique_urls = self.get_unique_parent_urls(categories_dict)
        
        if not unique_urls:
            logger.warning("No categories to process. Check if selected categories exist in the data.")
            return
        
        # Process all categories in parallel
        tasks = []
        for category, urls in unique_urls.items():
            logger.info(f"Queuing category: {category} with {len(urls)} URLs")
            task = self.process_category_with_semaphore(category, urls, max_urls_per_category)
            tasks.append(task)
        
        # Run all category processing tasks in parallel
        logger.info(f"Starting parallel processing of {len(tasks)} categories...")
        await asyncio.gather(*tasks)
        
        # Print final statistics
        self.print_statistics()
        
        logger.info("All categories completed. Parallel task generation finished.")


async def main():
    """Main function"""
    # Enable tracemalloc for better debugging
    tracemalloc.start()
    
    parser = argparse.ArgumentParser(description='Generate tasks for specific categories in parallel')
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=['shopping', 'services', 'health', 'travel', 'social', 'government', 'education', 'academic', 'entertainment', 'tech', 'news', 'food', 'finance'],
        default=['shopping', 'services', 'health', 'travel', 'social', 'government', 'education', 'academic', 'entertainment', 'tech', 'news', 'food', 'finance'],
        help='Specific categories to process (default: all)'
    )
    parser.add_argument('--max-urls-per-category', type=int, default=10000,
                       help='Maximum number of URLs to process per category (default: 10)')
    parser.add_argument('--max-concurrent-categories', type=int, default=20,
                       help='Maximum number of categories to process concurrently (default: 3)')
    
    args = parser.parse_args()
    
    # Load the expanded URL dictionary
    categories_file = "categories_dict_expand_V4.json"
    
    if not os.path.exists(categories_file):
        logger.error(f"Categories file not found: {categories_file}")
        return
    
    try:
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories_dict = json.load(f)
        
        logger.info(f"Loaded {len(categories_dict)} categories")
        
        if args.categories:
            logger.info(f"Selected categories: {args.categories}")
        
        # Initialize and run parallel task generator
        generator = ParallelTaskGenerator(
            selected_categories=args.categories,
            max_concurrent_categories=args.max_concurrent_categories
        )
        await generator.run(categories_dict, max_urls_per_category=args.max_urls_per_category)
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())
