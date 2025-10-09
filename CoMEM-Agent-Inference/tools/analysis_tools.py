"""Analysis tools for the GUI Agent framework"""
import json
import re
import time
from typing import Dict, Any, List, Optional
import sys
from urllib.parse import urljoin
sys.path.insert(0, 'GUI-Agent')
from bs4 import BeautifulSoup
import requests
from PIL import Image
import io
import base64
import torch
from .helpers import safe_download_image
from utils.training_data_collector import get_collector


class PageParserTool:
    """Tool for parsing and extracting content from web pages"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'page_parser'
        self.description = 'Get content of the current web page'
        self.parameters = {
            'type': 'object',
            'properties': {
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this parsing is necessary'
                }
            },
            'required': ['reasoning']
        }
    
    def call(self, page, **kwargs) -> str:
        """Parse the current page content"""
        try:
            # Get the current page URL
            url = page.url
            from qwen_agent.tools.web_extractor import SimpleDocParser
            # Use SimpleDocParser directly like web_extractor
            parsed_web = SimpleDocParser().call({'url': url})
            return parsed_web
            
        except Exception as e:
            return f"Error in page parser tool: {str(e)}"


class ImageCheckerTool:
    """Tool for analyzing images on the current page"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'image_checker'
        self.description = 'Get captions and analyze the first 5 images on the current page with AI-generated descriptions'
        self.parameters = {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Query or context for image analysis (e.g., "What products are shown?")'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why image analysis is necessary'
                }
            },
            'required': ['query', 'reasoning']
        }
        
        # Initialize CLIP model for image ranking
        self.clip_model = None
        self.clip_processor = None
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model initialized successfully")
        except Exception as e:
            print(f"Failed to initialize CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def call(self, args: str, **kwargs) -> str:
        """Analyze images on the current page with AI-generated descriptions"""
        try:
            if isinstance(args, str):
                args = json.loads(args)
            
            query = args.get('query', '')
            reasoning = args.get('reasoning', '')
            
            # Get the page from the environment
            page = kwargs.get('page')
            if not page:
                # Try to get page from trajectory if available
                trajectory = kwargs.get('trajectory')
                if trajectory and hasattr(trajectory, 'env') and hasattr(trajectory.env, 'page'):
                    page = trajectory.env.page
                else:
                    return "Error: No page context available"
            
            # Get page HTML with error handling
            html_content = ""
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    html_content = page.content()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5)  # Wait for navigation to complete
                        continue
                    else:
                        return f"Error: Could not retrieve page content after {max_retries} attempts: {e}"
        
            if not html_content:
                return "Error: No page content available"
                
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find all images
            images = soup.find_all('img')
            image_analysis = []
            
            if query and len(images) > 5 and self.clip_model and self.clip_processor:
                # Use CLIP to rank images by relevance to query
                try:
                    # Prepare image URLs and text query
                    image_urls = []
                    valid_images = []
                    
                    for img in images:
                        img_url = img.get('src', '')
                        if img_url:
                            # Convert relative URL to absolute if needed
                            if img_url.startswith('//'):
                                # Protocol-relative URL
                                img_url = f"https:{img_url}"
                            elif img_url.startswith('/'):
                                # Absolute path from domain root
                                img_url = urljoin(page.url, img_url)
                            elif not img_url.startswith('http'):
                                # Relative path
                                img_url = urljoin(page.url, img_url)
                            
                            # Skip data URLs and invalid URLs
                            if not img_url.startswith('data:') and img_url.startswith('http'):
                                image_urls.append(img_url)
                                valid_images.append(img)
                    print('valid image urls: ', len(image_urls), '/', len(images))    
                    
                    if image_urls and len(image_urls) > 5:
                        # Download and process images for CLIP using helper function
                        processed_images = []
                        valid_urls = []
                        
                        for img_url in image_urls:
                            response = safe_download_image(img_url)
                            if response:
                                try:
                                    image = Image.open(io.BytesIO(response.content))
                                    # Convert to RGB if needed
                                    if image.mode != 'RGB':
                                        image = image.convert('RGB')
                                    processed_images.append(image)
                                    valid_urls.append(img_url)
                                except Exception:
                                    continue
                        
                        if processed_images:
                            # Process images with CLIP
                            inputs = self.clip_processor(images=processed_images, text=[query] * len(processed_images), return_tensors="pt", padding=True)
                            
                            # Get CLIP embeddings
                            with torch.no_grad():
                                outputs = self.clip_model(**inputs)
                                logits_per_image = outputs.logits_per_image
                                probs = logits_per_image.softmax(dim=1)
                            
                            # Get similarity scores
                            similarities = probs[:, 0].tolist()  # Similarity to query
                            
                            # Sort images by similarity score
                            image_scores = list(zip(valid_images[:len(processed_images)], similarities, valid_urls))
                            image_scores.sort(key=lambda x: x[1], reverse=True)
                            
                            # Take top 5 most relevant images
                            top_images = image_scores[:5]
                            
                            for i, (img, score, img_url) in enumerate(top_images):
                                img_info = {
                                    'index': i + 1,
                                    'src': img_url,
                                    'alt': img.get('alt', ''),
                                    'title': img.get('title', ''),
                                    'width': img.get('width', ''),
                                    'height': img.get('height', ''),
                                    'caption': '',
                                    'relevance_score': round(score, 3)
                                }
                                
                                # Try to find caption in nearby text or figure elements
                                parent = img.parent
                                if parent:
                                    # Look for figcaption or caption-like elements
                                    caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                                    if caption_elem:
                                        img_info['caption'] = caption_elem.get_text(strip=True)
                                    else:
                                        # Look for text near the image
                                        nearby_text = parent.get_text(strip=True)
                                        if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                            img_info['caption'] = nearby_text
                                
                                image_analysis.append(img_info)
                        else:
                            # Fallback to first 5 images if CLIP processing fails
                            for i, img in enumerate(images[:5]):
                                img_info = {
                                    'index': i + 1,
                                    'src': img.get('src', ''),
                                    'alt': img.get('alt', ''),
                                    'title': img.get('title', ''),
                                    'width': img.get('width', ''),
                                    'height': img.get('height', ''),
                                    'caption': '',
                                    'relevance_score': None
                                }
                                
                                # Try to find caption in nearby text or figure elements
                                parent = img.parent
                                if parent:
                                    # Look for figcaption or caption-like elements
                                    caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                                    if caption_elem:
                                        img_info['caption'] = caption_elem.get_text(strip=True)
                                    else:
                                        # Look for text near the image
                                        nearby_text = parent.get_text(strip=True)
                                        if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                            img_info['caption'] = nearby_text
                                
                                image_analysis.append(img_info)
                    else:
                        # Not enough images or no query, use first 5
                        for i, img in enumerate(images[:5]):
                            img_info = {
                                'index': i + 1,
                                'src': img.get('src', ''),
                                'alt': img.get('alt', ''),
                                'title': img.get('title', ''),
                                'width': img.get('width', ''),
                                'height': img.get('height', ''),
                                'caption': '',
                                'relevance_score': None
                            }
                            
                            # Try to find caption in nearby text or figure elements
                            parent = img.parent
                            if parent:
                                # Look for figcaption or caption-like elements
                                caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                                if caption_elem:
                                    img_info['caption'] = caption_elem.get_text(strip=True)
                                else:
                                    # Look for text near the image
                                    nearby_text = parent.get_text(strip=True)
                                    if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                        img_info['caption'] = nearby_text
                            
                            image_analysis.append(img_info)
                            
                except Exception as e:
                    # Fallback to first 5 images if CLIP fails
                    for i, img in enumerate(images[:5]):
                        img_info = {
                            'index': i + 1,
                            'src': img.get('src', ''),
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'width': img.get('width', ''),
                            'height': img.get('height', ''),
                            'caption': '',
                            'relevance_score': None
                        }
                        
                        # Try to find caption in nearby text or figure elements
                        parent = img.parent
                        if parent:
                            # Look for figcaption or caption-like elements
                            caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                            if caption_elem:
                                img_info['caption'] = caption_elem.get_text(strip=True)
                            else:
                                # Look for text near the image
                                nearby_text = parent.get_text(strip=True)
                                if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                    img_info['caption'] = nearby_text
                        
                        image_analysis.append(img_info)
            else:
                # No query or not enough images, use first 5
                for i, img in enumerate(images[:5]):
                    img_info = {
                        'index': i + 1,
                        'src': img.get('src', ''),
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width', ''),
                        'height': img.get('height', ''),
                        'caption': '',
                        'relevance_score': None
                    }
                    
                    # Try to find caption in nearby text or figure elements
                    parent = img.parent
                    if parent:
                        # Look for figcaption or caption-like elements
                        caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                        if caption_elem:
                            img_info['caption'] = caption_elem.get_text(strip=True)
                        else:
                            # Look for text near the image
                            nearby_text = parent.get_text(strip=True)
                            if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                img_info['caption'] = nearby_text
                    
                    image_analysis.append(img_info)
                img_info = {
                    'index': i + 1,
                    'src': img.get('src', ''),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                    'caption': ''
                }
                
                # Try to find caption in nearby text or figure elements
                parent = img.parent
                if parent:
                    # Look for figcaption or caption-like elements
                    caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                    if caption_elem:
                        img_info['caption'] = caption_elem.get_text(strip=True)
                    else:
                        # Look for text near the image
                        nearby_text = parent.get_text(strip=True)
                        if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                            img_info['caption'] = nearby_text
                
                image_analysis.append(img_info)
            
            # Generate AI descriptions for images if LLM is available
            if hasattr(self, 'llm') and self.llm:
                for img_info in image_analysis:
                    try:
                        # Create multimodal message with image and query
                        image_url = img_info['src']
                        if image_url and not image_url.startswith('data:'):
                            # Convert relative URL to absolute if needed
                            if image_url.startswith('/'):
                                image_url = f"{page.url.rstrip('/')}{image_url}"
                            elif not image_url.startswith('http'):
                                image_url = f"{page.url.rstrip('/')}/{image_url.lstrip('/')}"
                        
                        # Prepare context for image description
                        context = f"Image context: {img_info.get('alt', '')} {img_info.get('title', '')} {img_info.get('caption', '')}"
                        if query:
                            context += f"\nQuery: {query}"
                        
                        # Create multimodal message
                        messages = [
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'type': 'text',
                                        'text': f"Please provide a short, descriptive caption {'related to the query' if query else ''} for this image. {context}"
                                    },
                                    {
                                        'type': 'image_url',
                                        'image_url': {
                                            'url': image_url
                                        }
                                    }
                                ]
                            }
                        ]
                        
                        # Call LLM for image description
                        response, _, _ = self.llm.chat(messages=messages, stream=False)
                        ai_description = response.content.strip()
                        # for resp in response:
                        #     if hasattr(resp, 'content'):
                        #         ai_description += resp.content
                        #     elif isinstance(resp, dict) and 'content' in resp:
                        #         # Handle dictionary response format
                        #         ai_description += resp['content']
                        #     else:
                        #         ai_description += str(resp)
                        
                        img_info['ai_description'] = ai_description
                        
                    except Exception as e:
                        img_info['ai_description'] = ""
            else:
                # If no LLM available, add placeholder
                for img_info in image_analysis:
                    img_info['ai_description'] = ""
            
            result = {
                'page_url': page.url,
                'total_images_found': len(images),
                'analyzed_images': image_analysis
            }
            
            return json.dumps(result, indent=2, ensure_ascii=False)
            
        except Exception as e:
            return f"Error in image checker tool: {str(e)}"


# @register_tool('map_search')
class MapSearchTool:
    """Tool for navigating to Google Maps for geographical searches"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'map_search'
        self.description = 'Navigate to Google Maps to search for geographical information'
        self.parameters = {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Search query for map information (e.g., "Nanning China location")'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this map search is necessary'
                }
            },
            'required': ['query', 'reasoning']
        }
    
    def call(self, args: str) -> str:
        """Navigate to Google Maps with the search query"""
        if isinstance(args, str):
            args = json.loads(args)
        
        query = args.get('query', '')
        reasoning = args.get('reasoning', '')

        # Navigate to Google Maps with the search query
        import urllib.parse
        encoded_query = urllib.parse.quote(query)
        google_maps_url = f"https://www.google.com/maps/search/{encoded_query}"
        
        return google_maps_url


# @register_tool('content_analyzer')
class ContentAnalyzerTool:
    """Tool for comprehensive page analysis using page parsing and image analysis"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'content_analyzer'
        self.description = 'Comprehensive page analysis including text content, images, and insights. Especially useful for information intensive pages.'
        self.parameters = {
            'type': 'object',
            'properties': {
                'query': {
                    'type': 'string',
                    'description': 'Query or context you want to find on the page (e.g., "What products are shown?")'
                },
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this content analysis is necessary'
                }
            },
            'required': ['query', 'reasoning']
        }
        
        # Initialize CLIP model for image analysis
        self.clip_model = None
        self.clip_processor = None
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            print("CLIP model initialized successfully for ContentAnalyzer")
        except Exception as e:
            print(f"Failed to initialize CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
        
        # Initialize LLM for content summarization
        self.llm = None
        if config and 'llm' in config:
            self.llm = config['llm']
    
    def _parse_page_content(self, page, query: str = "") -> str:
        """Internal method to parse page content using SimpleDocParser and LLM summarization"""
        # Try primary parser first
        try:
            # Get the current page URL
            url = page.url
            from qwen_agent.tools.web_extractor import SimpleDocParser
            # Use SimpleDocParser to extract content
            parsed_web = SimpleDocParser().call({'url': url})
        except Exception as e:
            # Fallback: try to get page content directly from Playwright and summarize
            try:
                try:
                    page.wait_for_load_state('domcontentloaded', timeout=3000)
                except Exception:
                    pass
                
                # Retry getting page content a few times
                html_content = ""
                for attempt in range(3):
                    try:
                        html_content = page.content()
                        if html_content:
                            break
                    except Exception:
                        time.sleep(0.5)
                if not html_content:
                    return f"Error parsing page content: {str(e)}"
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                for s in soup(["script", "style"]):
                    s.decompose()
                text = soup.get_text(separator=" ")
                parsed_web = ' '.join(text.split())
                
            except Exception as fallback_error:
                return f"Error parsing page content: {str(e)}. Fallback also failed: {str(fallback_error)}"
            
        # If primary parser succeeded, use LLM to extract useful info when available
        if hasattr(self, 'llm') and self.llm:
            content_to_summarize = parsed_web
            if query:
                system_prompt = (
                    "You are a helpful assistant that extracts and summarizes useful information from web page content. "
                    "Focus on information that is relevant to the user's specific query. Extract key facts, details, and insights that directly address what the user is looking for."
                )
                user_prompt = (
                    f"Query: {query}\n\nPlease extract and summarize information from this web page content that is relevant to the above query:\n\n{content_to_summarize}"
                )
            else:
                system_prompt = (
                    "You are a helpful assistant that extracts and summarizes useful information from web page content. "
                    "Focus on the main topics, key facts, and important details. Remove noise and irrelevant information."
                )
                user_prompt = f"Please extract and summarize the useful information from this web page content:\n\n{content_to_summarize}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response, _, _ = self.llm.chat(messages=messages, stream=False)
            summary = response.content.strip()
            return summary
        else:
            # No LLM available, return truncated content
            if isinstance(parsed_web, str):
                return parsed_web[:1000] + "..." if len(parsed_web) > 1000 else parsed_web
            else:
                return str(parsed_web)[:1000] + "..."
                    
        
    def _analyze_images(self, page, query: str) -> dict:
        """Internal method to analyze images using CLIP"""
        # try:
        # Get page HTML with error handling
        html_content = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                html_content = page.content()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.5)
                    continue
                else:
                    return {"error": f"Could not retrieve page content after {max_retries} attempts: {e}"}
        
        if not html_content:
            return {"error": "No page content available"}
        
        soup = BeautifulSoup(html_content, 'html.parser')
        images = soup.find_all('img')
        image_analysis = []
        
        if query and len(images) > 5 and self.clip_model and self.clip_processor:
            # Use CLIP to rank images by relevance to query
            # try:
            # Prepare image URLs and text query
            image_urls = []
            valid_images = []
            
            for img in images:
                img_url = img.get('src', '')
                if img_url:
                    # Convert relative URL to absolute if needed
                    if img_url.startswith('//'):
                        # Protocol-relative URL
                        img_url = f"https:{img_url}"
                    elif img_url.startswith('/'):
                        # Absolute path from domain root
                        img_url = urljoin(page.url, img_url)
                    elif not img_url.startswith('http'):
                        # Relative path
                        img_url = urljoin(page.url, img_url)
                    
                    # Skip data URLs and invalid URLs
                    if not img_url.startswith('data:') and img_url.startswith('http'):
                        image_urls.append(img_url)
                        valid_images.append(img)
            print('valid image urls: ', len(image_urls), '/', len(images))    
            if image_urls and len(image_urls) > 5:
                # Download and process images for CLIP using helper function
                processed_images = []
                valid_urls = []
                
                for img_url in image_urls:
                    response = safe_download_image(img_url)
                    if response:
                        try:
                            image = Image.open(io.BytesIO(response.content))
                            # Convert to RGB if needed
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            processed_images.append(image)
                            valid_urls.append(img_url)
                        except Exception:
                            continue
                
                if processed_images:
                    # Process images with CLIP
                    inputs = self.clip_processor(images=processed_images, text=[query] * len(processed_images), return_tensors="pt", padding=True)
                    
                    # Get CLIP embeddings
                    with torch.no_grad():
                        outputs = self.clip_model(**inputs)
                        logits_per_image = outputs.logits_per_image
                        probs = logits_per_image.softmax(dim=1)
                    
                    # Get similarity scores
                    similarities = probs[:, 0].tolist()  # Similarity to query
                    
                    # Sort images by similarity score
                    image_scores = list(zip(valid_images[:len(processed_images)], similarities, valid_urls))
                    image_scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Take top 5 most relevant images
                    top_images = image_scores[:5]
                    
                    for i, (img, score, img_url) in enumerate(top_images):
                        img_info = {
                            'index': i + 1,
                            'src': img_url,
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'width': img.get('width', ''),
                            'height': img.get('height', ''),
                            'caption': '',
                            'relevance_score': round(score, 3)
                        }
                        
                        # Try to find caption in nearby text or figure elements
                        parent = img.parent
                        if parent:
                            # Look for figcaption or caption-like elements
                            caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                            if caption_elem:
                                img_info['caption'] = caption_elem.get_text(strip=True)
                            else:
                                # Look for text near the image
                                nearby_text = parent.get_text(strip=True)
                                if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                    img_info['caption'] = nearby_text
                        
                        image_analysis.append(img_info)
                else:
                    # Fallback to first 5 images if CLIP processing fails
                    for i, img in enumerate(images[:5]):
                        img_info = {
                            'index': i + 1,
                            'src': img.get('src', ''),
                            'alt': img.get('alt', ''),
                            'title': img.get('title', ''),
                            'width': img.get('width', ''),
                            'height': img.get('height', ''),
                            'caption': '',
                            'relevance_score': None
                        }
                        
                        # Try to find caption in nearby text or figure elements
                        parent = img.parent
                        if parent:
                            # Look for figcaption or caption-like elements
                            caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                            if caption_elem:
                                img_info['caption'] = caption_elem.get_text(strip=True)
                            else:
                                # Look for text near the image
                                nearby_text = parent.get_text(strip=True)
                                if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                    img_info['caption'] = nearby_text
                        
                        image_analysis.append(img_info)
            else:
                # Not enough images or no query, use first 5
                for i, img in enumerate(images[:5]):
                    img_info = {
                        'index': i + 1,
                        'src': img.get('src', ''),
                        'alt': img.get('alt', ''),
                        'title': img.get('title', ''),
                        'width': img.get('width', ''),
                        'height': img.get('height', ''),
                        'caption': '',
                        'relevance_score': None
                    }
                    
                    # Try to find caption in nearby text or figure elements
                    parent = img.parent
                    if parent:
                        # Look for figcaption or caption-like elements
                        caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                        if caption_elem:
                            img_info['caption'] = caption_elem.get_text(strip=True)
                        else:
                            # Look for text near the image
                            nearby_text = parent.get_text(strip=True)
                            if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                                img_info['caption'] = nearby_text
                    
                    image_analysis.append(img_info)
                    
                # except Exception as e:
                #     # Fallback to first 5 images if CLIP fails
                #     for i, img in enumerate(images[:5]):
                #         img_info = {
                #             'index': i + 1,
                #             'src': img.get('src', ''),
                #             'alt': img.get('alt', ''),
                #             'title': img.get('title', ''),
                #             'width': img.get('width', ''),
                #             'height': img.get('height', ''),
                #             'caption': '',
                #             'relevance_score': None
                #         }
                        
                #         # Try to find caption in nearby text or figure elements
                #         parent = img.parent
                #         if parent:
                #             # Look for figcaption or caption-like elements
                #             caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                #             if caption_elem:
                #                 img_info['caption'] = caption_elem.get_text(strip=True)
                #             else:
                #                 # Look for text near the image
                #                 nearby_text = parent.get_text(strip=True)
                #                 if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                #                     img_info['caption'] = nearby_text
                        
                #         image_analysis.append(img_info)
        else:
            # No query or not enough images, use first 5
            for i, img in enumerate(images[:5]):
                img_info = {
                    'index': i + 1,
                    'src': img.get('src', ''),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width', ''),
                    'height': img.get('height', ''),
                    'caption': '',
                    'relevance_score': None
                }
                
                # Try to find caption in nearby text or figure elements
                parent = img.parent
                if parent:
                    # Look for figcaption or caption-like elements
                    caption_elem = parent.find('figcaption') or parent.find(class_=re.compile(r'caption|description', re.I))
                    if caption_elem:
                        img_info['caption'] = caption_elem.get_text(strip=True)
                    else:
                        # Look for text near the image
                        nearby_text = parent.get_text(strip=True)
                        if nearby_text and len(nearby_text) < 200:  # Reasonable caption length
                            img_info['caption'] = nearby_text
                
                image_analysis.append(img_info)
        
        # Generate AI descriptions for images if LLM is available
        if hasattr(self, 'llm') and self.llm:
            for img_info in image_analysis:
                # Create multimodal message with image and query
                image_url = img_info['src']
                if image_url and not image_url.startswith('data:'):
                    # Convert relative URL to absolute if needed
                    if image_url.startswith('/'):
                        image_url = f"{page.url.rstrip('/')}{image_url}"
                    elif not image_url.startswith('http'):
                        image_url = f"{page.url.rstrip('/')}/{image_url.lstrip('/')}"
                
                # Prepare context for image description
                context = f"Image context: {img_info.get('alt', '')} {img_info.get('title', '')} {img_info.get('caption', '')}"
                if query:
                    context += f"\nQuery: {query}"
                
                # Create multimodal message using Qwen-Agent schema
                from qwen_agent.llm.schema import ContentItem, Message
                
                # Download and convert image to base64
                try:
                    img_response = requests.get(image_url, timeout=10)
                    if img_response.status_code == 200:
                        img_data = img_response.content
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        image_data = f"data:image/png;base64,{img_base64}"
                    else:
                        # If download fails, skip this image
                        continue
                except Exception as e:
                    print(f"Failed to download image {image_url}: {e}")
                    continue
                
                messages = [
                    {
                        'role': 'user',
                        'content': [
                            {
                                "type": "text",
                                "text": f"Please provide a short, descriptive caption {'related to the query' if query else ''} for this image. {context}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data
                                }
                            }
                        ]
                    }
                ]
                
                # Call LLM for image description
                ai_description = ""
                try:
                    response, _, _ = self.llm.chat(messages=messages, stream=False)
                    ai_description = response.content.strip()
                    # for resp in response:
                    #     if hasattr(resp, 'content'):
                    #         ai_description += resp.content
                    #     elif isinstance(resp, dict) and 'content' in resp:
                    #         # Handle dictionary response format
                    #         ai_description += resp['content']
                    #     else:
                    #         ai_description += str(resp)
                except Exception as e:
                    print(f"Failed to generate image description for {image_url}: {e}")
                    continue
                
                img_info['ai_description'] = ai_description
                
        else:
            # If no LLM available, add placeholder
            for img_info in image_analysis:
                img_info['ai_description'] = ""
        
        return {
            'page_url': page.url,
            'total_images_found': len(images),
            'analyzed_images': image_analysis
        }
            
        # except Exception as e:
        #     return {"error": f"Error in image analysis: {str(e)}"}
    
    def call(self, args: str, **kwargs) -> str:
        """Comprehensive page analysis combining text and image analysis"""
        # try:
        if isinstance(args, str):
            args = json.loads(args)
        
        query = args.get('query', '')
        reasoning = args.get('reasoning', '')
        analyze_images = args.get('analyze_images', True)
        # Get the page from the environment
        page = kwargs.get('page')
        if not page:
            # Try to get page from trajectory if available
            trajectory = kwargs.get('trajectory')
            if trajectory and hasattr(trajectory, 'env') and hasattr(trajectory.env, 'page'):
                page = trajectory.env.page
            else:
                return "Error: No page context available"
        
        # Step 1: Parse page content
        page_content = self._parse_page_content(page, query)
        print('page_content: ', page_content)
        # Step 2: Analyze images
        if analyze_images:
            image_analysis = self._analyze_images(page, query)
            print('image_analysis: ', image_analysis)
        else:
            image_analysis = None
            
        # Step 3: Generate comprehensive analysis
        analysis = {
            'page_url': page.url,
            'query': query,
            'page_content': page_content,
            'image_analysis': image_analysis,
            'summary': self._generate_summary(page_content, image_analysis, query)
        }
        
        return json.dumps(analysis, indent=2, ensure_ascii=False)
            
        # except Exception as e:
        #     return f"Error in content analyzer tool: {str(e)}"
    
    def _generate_summary(self, page_content: str, image_analysis: dict, query: str) -> str:
        """Generate a comprehensive summary based on page content and image analysis"""
        summary_parts = []
        
        # Add page content summary
        if page_content and not page_content.startswith("Error"):
            summary_parts.append(f"Page Content: {page_content}")
        
        # Add image analysis summary
        if 'analyzed_images' in image_analysis and image_analysis['analyzed_images']:
            images = image_analysis['analyzed_images']
            summary_parts.append(f"Found {len(images)} relevant images:")
            img_analysis = ''
            for img in images:
                img_analysis += f"- Image {img['index']}: {img.get('ai_description', 'No description available')}\n"
            summary_parts.append(img_analysis)
        
        # Generate query-specific insights using LLM if available
        if query and hasattr(self, 'llm') and self.llm:
            # Prepare the combined content for LLM analysis
            combined_content = f"Query: {query}\n\n"
            combined_content += f"Page Content Summary: {page_content}\n\n"
            
            if 'analyzed_images' in image_analysis and image_analysis['analyzed_images']:
                combined_content += "Image Analysis:\n"
                for img in image_analysis['analyzed_images']:
                    if 'ai_description' in img and img['ai_description']:
                        combined_content += f"- Image {img['index']}: {img['ai_description']}\n"
            
            # Create prompt for LLM to generate insights
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes web page content and images to provide query-specific insights. Focus on answering the user's specific question and highlighting the most relevant information found."
                },
                {
                    "role": "user",
                    "content": f"Based on the following page content and image analysis, provide 2-3 key insights that directly answer this query: '{query}'\n\n{combined_content}"
                }
            ]
            
            # Get LLM response for insights
            response, _, _ = self.llm.chat(messages=messages, stream=False)
            insights = ""
            for resp in response:
                if hasattr(resp, 'content'):
                    insights += resp.content
                elif isinstance(resp, dict) and 'content' in resp:
                    # Handle dictionary response format
                    insights += resp['content']
                else:
                    insights += str(resp)
            
            summary_parts.append(f"Query-Specific Insights: {insights.strip()}")
                
        return "\n".join(summary_parts) 


# @register_tool('goto_homepage')
class GotoHomepageTool:
    """Tool for navigating to specific pages based on user intent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.name = 'goto_homepage'
        self.description = 'Navigate to the homepage with all available websites'
        self.parameters = {
            'type': 'object',
            'properties': {
                'reasoning': {
                    'type': 'string',
                    'description': 'Reasoning for why this navigation is necessary'
                }
            },
            'required': ['reasoning']
        }
        
        # Define the mapping of page names to URLs (restricted set)
        self.page_urls = {
            'homepage': 'http://localhost:8080/'
        }
        
        # Define actions that are considered complete after clicking search
        self.complete_after_search = {
            'homepage'
        }
    
    def call(self, args: str) -> str:
        """Return the target URL for the specified page based on user intent"""
        if isinstance(args, str):
            args = json.loads(args)
        
        reasoning = args.get('reasoning', '')
        return 'http://localhost:8080/'