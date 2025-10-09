import json
import re
import ast
import requests
import aiohttp
import asyncio
import os
from typing import Dict, Any, List
from qwen_agent.tools import BaseTool
from qwen_agent.tools.base import register_tool
from .helpers import get_random_headers
from playwright.async_api import async_playwright

# Configuration Constants
SERPAPI_API_KEY = ""
SERPAPI_URL = "https://serpapi.com/search"

# Available search engines
SEARCH_ENGINES = {
    "google": "google",
    "bing": "bing", 
    "duckduckgo": "duckduckgo",
    "yahoo": "yahoo",
    "baidu": "baidu"
}

@register_tool('google_web_search')
class WebSearchTool(BaseTool):
    """Tool for performing web searches and extracting relevant information"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.llm = None  # Will be set by the agent
        self.max_queries = 3
        self.max_results_per_query = 3
        self.search_engines = ["google", "bing"]  # Default engines to use
        
        # Tool metadata for function calling
        self.description = "Search the web using multiple search engines (Google, Bing, etc.) and extract relevant content. Use this when you need to find information that's not available on the current page."
        self.parameters = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find information about"
                },
                "reasoning": {
                    "type": "string", 
                    "description": "Why you need to search for this information"
                }
            },
            "required": ["query", "reasoning"]
        }
        
    def set_llm(self, llm):
        """Set the LLM instance from the main agent"""
        self.llm = llm
    
    def set_search_engines(self, engines: List[str]):
        """Set which search engines to use"""
        valid_engines = []
        for engine in engines:
            if engine in SEARCH_ENGINES:
                valid_engines.append(engine)
            else:
                print(f"Warning: Unknown search engine '{engine}'. Available: {list(SEARCH_ENGINES.keys())}")
        
        if valid_engines:
            self.search_engines = valid_engines
            print(f"Using search engines: {self.search_engines}")
        else:
            print("No valid engines provided, using default: google")
            self.search_engines = ["google"]
        
    def _call_llm(self, messages):
        """Call the LLM with proper error handling"""
        try:
            response, _, _ = self.llm.chat(messages=messages, stream=False)
            result = ""
            for resp in response:
                result += getattr(resp, 'content', str(resp))
            return result.strip()
        except Exception as e:
            print(f"Error calling LLM in web search: {e}")
            return None
    
    def _generate_search_queries(self, user_query: str, number_of_queries: int = 3) -> List[str]:
        """Generate search queries based on user query"""
        prompt = (
            f"You are an expert information gathering assistant. "
            f"Given the user's query, generate up to {number_of_queries} distinct, precise search queries that would help gather comprehensive "
            "information on the topic. The queries should help find relevant and useful information. "
            "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
        )
        messages = [
            {"role": "system", "content": "You are a helpful and precise research assistant."},
            {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
        ]
        response = self._call_llm(messages)
        if response:
            try:
                match = re.search(r"\[.*?\]", response, re.DOTALL)
                if match:
                    list_str = match.group(0)
                    search_queries = ast.literal_eval(list_str)
                    if isinstance(search_queries, list):
                        return search_queries[:number_of_queries]
            except Exception as e:
                print(f"Error parsing search queries: {e}")
        return []
    
    def _perform_search(self, query: str) -> List[str]:
        """Perform a search using multiple engines via SERPAPI and return URLs"""
        all_links = []
        
        for engine in self.search_engines:
            if engine not in SEARCH_ENGINES:
                print(f"Unknown search engine: {engine}")
                continue
                
            params = {
                "q": query,
                "api_key": SERPAPI_API_KEY,
                "engine": SEARCH_ENGINES[engine],
                "num": self.max_results_per_query  # Control number of results per engine
            }
            
            try:
                print(f"Searching with {engine}...")
                response = requests.get(SERPAPI_URL, params=params, timeout=10)
                if response.status_code == 200:
                    results = response.json()
                    if "organic_results" in results:
                        links = [item.get("link") for item in results["organic_results"] if "link" in item]
                        all_links.extend(links[:self.max_results_per_query])
                        print(f"Found {len(links[:self.max_results_per_query])} results from {engine}")
                    else:
                        print(f"No organic results from {engine}")
                else:
                    print(f"SERPAPI error for {engine}: {response.status_code}")
            except Exception as e:
                print(f"Error performing {engine} search: {e}")
        
        # Remove duplicates while preserving order
        unique_links = []
        seen = set()
        for link in all_links:
            if link not in seen:
                unique_links.append(link)
                seen.add(link)
        
        print(f"Total unique results: {len(unique_links)}")
        return unique_links[:self.max_results_per_query * len(self.search_engines)]
    
    def _is_page_useful(self, user_query: str, page_text: str, screenshot_path: str = None) -> bool:
        """Determine if a page is useful for the user query, optionally considering screenshot"""
        if screenshot_path and os.path.exists(screenshot_path):
            # Split screenshot into 3 subfigures
            subfigure_paths = self._split_screenshot_into_subfigures(screenshot_path)
            
            # Use multimodal evaluation with 3 subfigures
            prompt = (
                f"Given the following user task: '{user_query}', the webpage content, and 3 sections of a webpage screenshot (top, middle, bottom), "
                "determine if the page is useful for the task. Consider both the text content and visual elements from all sections. "
                "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not."
            )
            
            # Create multimodal message with 3 subfigures
            from qwen_agent.llm.schema import ContentItem
            
            content_items = [
                ContentItem(text=f"User Query: {user_query}\n\nWebpage Content (first 10000 characters):\n{page_text[:10000]}\n\n{prompt}")
            ]
            
            # Add each subfigure as a separate image
            for i, subfigure_path in enumerate(subfigure_paths):
                if os.path.exists(subfigure_path):
                    # Convert image file to base64
                    import base64
                    with open(subfigure_path, 'rb') as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        image_data = f"data:image/png;base64,{img_base64}"
                    
                    content_items.append(ContentItem(image=image_data))
            
            messages = [
                {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
                {"role": "user", "content": content_items}
            ]
        else:
            # Text-only evaluation
            prompt = (
                f"Given the following user task: '{user_query}', and the webpage content, determine if the page is useful for the task. "
                "Consider relevance, depth, credibility, and whether it provides informative content. "
                "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not."
            )
            messages = [
                {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
                {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 10000 characters):\n{page_text[:10000]}\n\n{prompt}"}
            ]
        
        response = self._call_llm(messages)
        if response:
            answer = response.strip().lower()
            return answer == "yes"
        return False
    
    def _extract_relevant_context(self, user_query: str, search_query: str, page_text: str) -> str:
        """Extract relevant context from page text"""
        prompt = (
            f"You are an expert information extractor. The user's original query is: '{user_query}' and the search query that led to this page is: '{search_query}'. "
            "Given the user's query, the search query that led to this page, and the webpage content, "
            "extract all pieces of information that are relevant to answering the user's query. "
            "Return only the relevant context as plain text without commentary."
        )
        messages = [
            {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
            {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 10000 characters):\n{page_text[:10000]}\n\n{prompt}"}
        ]
        response = self._call_llm(messages)
        return response.strip() if response else ""
    
    def _fetch_page_content(self, url: str) -> str:
        """Fetch page content from URL and parse HTML to extract clean text"""
        try:
            headers = get_random_headers()
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html_content = response.text
                
                # Parse HTML to extract clean text
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Remove script, style, and navigation elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()
                
                # Extract text content
                text = soup.get_text(separator=' ', strip=True)
                
                # Clean up whitespace and normalize
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = ' '.join(chunk for chunk in chunks if chunk)
                
                return clean_text
            else:
                print(f"Failed to fetch page content: {response.status_code}")
                return ""
        except Exception as e:
            print(f"Error fetching page content: {e}")
            return ""
    
    def _safe_filename(self, url: str) -> str:
        """Create a safe filename from URL"""
        name = url.replace("https://", "").replace("http://", "")
        # Replace all non-alphanumeric, dot, dash, or underscore with _
        name = re.sub(r'[^A-Za-z0-9._-]', '_', name)
        return name[:100] + ".png"
    
    async def _take_screenshot_async(self, url: str, output_dir: str = "screenshots") -> str:
        """Take screenshot of a webpage"""
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, self._safe_filename(url))
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--start-maximized"
            ])
            context = await browser.new_context(
                user_agent=get_random_headers()['User-Agent'],
                viewport={"width": 1920, "height": 1080},
                locale="en-US"
            )
            page = await context.new_page()
            await page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            try:
                await page.goto(url, timeout=20000)
                await asyncio.sleep(2)
                await page.screenshot(path=filename, full_page=True)
                print(f"Screenshot saved to {filename}")
                return filename
            except Exception as e:
                print(f"Failed to take screenshot: {e}")
                return None
            finally:
                await browser.close()
    
    def _split_screenshot_into_subfigures(self, screenshot_path: str) -> List[str]:
        """Split a screenshot into 3 subfigures (top, middle, bottom)"""
        try:
            from PIL import Image
            
            # Open the screenshot
            img = Image.open(screenshot_path)
            width, height = img.size
            
            # Calculate heights for each section (top, middle, bottom)
            section_height = height // 3
            
            subfigure_paths = []
            base_name = screenshot_path.replace('.png', '')
            
            for i in range(3):
                # Calculate crop box for each section
                top = i * section_height
                bottom = (i + 1) * section_height if i < 2 else height  # Last section gets remaining pixels
                
                # Crop the image
                section = img.crop((0, top, width, bottom))
                
                # Save the subfigure
                subfigure_path = f"{base_name}_part{i+1}.png"
                section.save(subfigure_path)
                subfigure_paths.append(subfigure_path)
                # print(f"Subfigure {i+1} saved: {subfigure_path}")
            
            return subfigure_paths
            
        except Exception as e:
            print(f"Error splitting screenshot: {e}")
            return [screenshot_path]  # Return original if splitting fails
    
    async def call(self, args: str, **kwargs) -> str:
        """Main method to perform web search and extract relevant information"""
        try:
            # Parse arguments
            if isinstance(args, str):
                args_dict = json.loads(args)
            else:
                args_dict = args
            
            user_query = args_dict.get('query', '')
            if not user_query:
                return "Error: No query provided"
            
            # Generate search queries
            search_queries = self._generate_search_queries(user_query, self.max_queries)
            if not search_queries:
                return "Error: Could not generate search queries"
            
            print(f"Generated search queries: {search_queries}")
            
            # Perform searches
            all_links = []
            for query in search_queries:
                links = self._perform_search(query)
                for link in links:
                    if link not in [l['url'] for l in all_links]:
                        all_links.append({'url': link, 'query': query})
            
            print(f"Found {len(all_links)} unique links")
            
            # Process each link (only take screenshot for first one)
            useful_results = []
            first_screenshot_path = None
            
            for i, link_info in enumerate(all_links[:3]):
                url = link_info['url']
                query = link_info['query']
                print(f"********************** Processing {i+1}: {url} **********************")
                
                # 1. Fetch and parse page content
                page_text = self._fetch_page_content(url)
                if not page_text:
                    print(f"Skipping {url} - no content fetched")
                    continue
                
                # 2. Extract relevant context first
                context = self._extract_relevant_context(user_query, query, page_text)
                print(f"Context: {context}")
                if not context:
                    print(f"Skipping {url} - no relevant context extracted")
                    continue
                
                # 3. Take screenshot only for the first page
                screenshot_path = await self._take_screenshot_async(url)
                
                # 4. Determine if page is useful (use first screenshot for all pages)
                # if self._is_page_useful(user_query, context, screenshot_path):
                    # print(f"Page is useful - adding to results")
                useful_results.append({
                    'url': url,
                    'query': query,
                    'context': context,
                    'screenshot': screenshot_path
                })
                # else:
                #     print(f"Page not useful - skipping")
            
            
            # Return the extracted contexts directly
            if useful_results:
                # Combine all extracted contexts with source URLs and screenshot info
                result_parts = []
                for result in useful_results:
                    part = f"Source: {result['url']}\n{result['context']}"
                    if result.get('screenshot'):
                        part += f"\n[Screenshot available: {result['screenshot']}]"
                    result_parts.append(part)
                
                result_text = "\n\n".join(result_parts)
                return result_text
            else:
                return "No useful information found for the query."
                
        except Exception as e:
            return f"Error in web search: {str(e)}" 