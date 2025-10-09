# Tools Module

Function calling tools for the GUI agent to extend its capabilities.

## Components

### GUI Interaction Tools (`gui_tools.py`)

Basic browser interaction tools registered for function calling:
- **ClickTool**: Click on elements
- **TypeTool**: Type text into input fields
- **ScrollTool**: Scroll page (up, down, left, right)
- **SelectionTool**: Select from dropdown menus
- **WaitTool**: Wait for page loading
- **PressKeyTool**: Press special keys (enter, delete, space)
- **StopTool**: Complete task with answer
- **PageGotoTool**: Navigate to specific pages/websites

### Analysis Tools (`analysis_tools.py`)

Advanced content analysis tools:
- **PageParserTool**: Extract and parse web page content
- **ImageCheckerTool**: Analyze images with CLIP-based ranking
- **MapSearchTool**: Navigate to Google Maps
- **ContentAnalyzerTool**: Comprehensive page analysis (text + images)
- **GotoHomepageTool**: Navigate to homepage

### Web Search Tools (`web_search_tools.py`)

External information retrieval:
- **WebSearchTool**: Multi-engine web search (Google, Bing, DuckDuckGo)
  - Query generation and execution
  - Result aggregation
  - Content extraction

### Helper Functions (`helpers.py`)

Utility functions for tools:
- HTTP request handling
- Image downloading
- Error handling

## Tool Usage

Tools are automatically registered and available to the agent via function calling:

```python
# Agent automatically uses tools based on task requirements
# Example tool call from agent:
{
    "name": "click",
    "arguments": {
        "element_id": "123",
        "coords": "<point>100 200</point>",
        "description": "search button",
        "reasoning": "Need to submit search query"
    }
}
```

## Tool Features

### GUI Tools
- Natural language element descriptions
- Coordinate-based interaction
- Reasoning traces for each action

### Analysis Tools
- CLIP-based image ranking
- LLM-powered content summarization
- Multimodal page understanding

### Search Tools
- Multiple search engine support
- Async parallel queries
- Smart result filtering

## Configuration

Some tools require additional setup:
- **WebSearchTool**: Optionally configure SERPAPI key for enhanced results
- **ImageCheckerTool**: Requires CLIP model (auto-downloaded)
- **ContentAnalyzerTool**: Benefits from LLM for summarization

## Notes

- Tools follow Qwen-Agent's `BaseTool` interface
- All tools support reasoning/explanation parameters
- Tools return structured JSON responses
- Error handling included for robustness

