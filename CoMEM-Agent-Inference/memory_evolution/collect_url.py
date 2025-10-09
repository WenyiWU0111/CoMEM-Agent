import os
import json

config_paths = ['GUI-Agent/Mind2Web/evaluation_data/test_website',
                'GUI-Agent/Mind2Web/evaluation_data/test_domain_Service',
                'GUI-Agent/Mind2Web/evaluation_data/test_domain_Info',
                'GUI-Agent/webvoyager_evaluation/data/Allrecipes',
                'GUI-Agent/webvoyager_evaluation/data/Amazon',
                'GUI-Agent/webvoyager_evaluation/data/Apple',
                'GUI-Agent/webvoyager_evaluation/data/ArXiv',
                'GUI-Agent/webvoyager_evaluation/data/Booking',
                'GUI-Agent/webvoyager_evaluation/data/GitHub',
                'GUI-Agent/webvoyager_evaluation/data/Google_Map',
                'GUI-Agent/webvoyager_evaluation/data/Google_Search',
                'GUI-Agent/webvoyager_evaluation/data/Google_Flights',
                'GUI-Agent/webvoyager_evaluation/data/ESPN',
                'GUI-Agent/webvoyager_evaluation/data/Huggingface',
                'GUI-Agent/webvoyager_evaluation/data/BBC_News',
                'GUI-Agent/webvoyager_evaluation/data/ESPN',
                'GUI-Agent/webvoyager_evaluation/data/Wolfram_Alpha']
website_set = {}

for config_path in config_paths:
    all_files = os.listdir(config_path)
    for file in all_files:
        with open(os.path.join(config_path, file), 'r') as f:
            config = json.load(f)
            url = config['start_url']
            intent = config['intent']
            if url not in website_set:
                website_set[url] = []
            if len(website_set[url]) < 50:
                website_set[url].append(intent)

print(len(website_set))


from agent.llm_config import load_tool_llm
from browser_env.envs import ScriptBrowserEnv
from config.argument_parser import config
from PIL import Image
import base64
from io import BytesIO

tool_llm = load_tool_llm(config())
browser_env = ScriptBrowserEnv()

categories = [
  "shopping",
  "travel",
  "academic",
  "news",
  "government",
  "health",
  "entertainment",
  "finance",
  "food",
  "tech",
  "education",
  "career",
  "social",
  "media",
  "sports",
  "services",
  "other",
]

categories_dict = {}

categorize_prompt = """
You are a helpful assistant that categorizes websites into different categories.
You will be given a website URL and a list of categories.
You need to categorize the website into the most appropriate category.

Here is the website URL: {url}
Here is the list of categories: {categories}

Please return the category that the website belongs to.
Your response should be in the following format:
<category>category</category>
Your response should exactly be in the given categories! The category should be a single word! Do not need any explanation!
"""

for url, intents in website_set.items():
    # browser_env.page.goto(url)
    # screenshot = browser_env.page.screenshot()
    # screenshot_img = Image.open(BytesIO(screenshot))
    # buffered = BytesIO()
    # screenshot_img.save(buffered, format="PNG")
    # img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    # base64_image = f"data:image/png;base64,{img_str}"
    messages = [
        {"role": "user", "content": categorize_prompt.format(url=url, categories=categories)}
        # {"type": "image_url", "image_url": {"url": base64_image}}
    ]
    
    response, _, _ = tool_llm.chat(messages).content
    response = response.split('<category>')[-1].split('</category>')[0]
    print(response)
    categories_dict.setdefault(response, []).append({url: intents})

print(categories_dict)

with open('categories_dict.json', 'w') as f:
    json.dump(categories_dict, f)