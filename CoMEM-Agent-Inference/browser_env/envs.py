"""Simplified browser environment for image-only observations"""
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from beartype import beartype
from gymnasium import Env
from gymnasium.spaces import Box
from playwright.sync_api import (
    CDPSession,
    Page,
    Playwright,
    ViewportSize,
    sync_playwright,
)

from .actions import Action
from .action_parser_ground import execute_pixel_action
from .processors import SimpleImageObservationProcessor, SimpleTextObservationProcessor
from .utils import (
    DetachedPage,
    Observation,
)

class ScriptBrowserEnv(Env[dict[str, Observation], Action]):
    """
    Simplified browser environment that only supports image observations.
    The observation space is the current page screenshot.
    """

    @beartype
    def __init__(
        self,
        headless: bool = True,
        slow_mo: int = 0,
        viewport_size: ViewportSize = {"width": 1280, "height": 720},
        save_trace_enabled: bool = False,
        sleep_after_execution: float = 0.0,
        args = None,  # Additional arguments for the environment
    ):
        self.headless = headless
        self.slow_mo = slow_mo
        self.reset_finished = False
        self.viewport_size = viewport_size
        self.save_trace_enabled = save_trace_enabled
        self.sleep_after_execution = sleep_after_execution
        self.args = args
        self.tracing_started = False  # Track if tracing was started

        # Initialize observation processors for image and text observations
        self.image_processor = SimpleImageObservationProcessor(args)
        self.text_processor = SimpleTextObservationProcessor(args)
        from tools.analysis_tools import PageParserTool
        self.page_parser = PageParserTool()

        # Initialize Playwright
        self.playwright: Playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self.context_manager = self.browser.new_context(
            viewport=self.viewport_size,
            record_video_dir=None,
        )
        self.context = self.context_manager.__enter__()
        self.page: Page = self.context.new_page()

        # Setup observation space
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.viewport_size["height"], self.viewport_size["width"], 3),
            dtype=np.uint8,
        )

    @beartype
    def setup(self, config_file: Path | None = None) -> None:
        """Setup the browser environment"""
        if config_file is not None:
            with open(config_file, "r") as f:
                config = json.load(f)

            # Navigate to the specified URL
            if "start_url" in config:
                if '7770' in config['start_url']:
                    config['start_url'] = "http://ec2-3-146-212-252.us-east-2.compute.amazonaws.com:7770/"
                if 'wikipedia' in config['start_url']:
                    config['start_url'] = "https://www.wikipedia.org/"
                print(f"Navigating to {config['start_url']}")
                
                try:
                    self.page.goto(config["start_url"], timeout=60000)
                except Exception:
                    try:
                        self.page.goto(config["start_url"], timeout=60000)
                    except Exception as e:
                        print(f"Error navigating to {config['start_url']}: {e}")
                        try:
                            self.page.goto("about:blank")
                        except Exception:
                            pass
                        
                if 'sites' in config and "gitlab" in config['sites']:
                    from .env_config import ACCOUNTS
                    username = ACCOUNTS["gitlab"]["username"]
                    password = ACCOUNTS["gitlab"]["password"]
                    self.page.goto(f"{config['start_url']}/users/sign_in")
                    self.page.get_by_test_id("username-field").click()
                    self.page.get_by_test_id("username-field").fill(username)
                    self.page.get_by_test_id("username-field").press("Tab")
                    self.page.get_by_test_id("password-field").fill(password)
                    self.page.get_by_test_id("sign-in-button").click(timeout=60000)
            else:
                # Default to a blank page
                print("Navigating to about:blank")
                self.page.goto("about:blank")
        else:
            # Default to a blank page
            self.page.goto("about:blank")
        
        # Start tracing if enabled
        if self.save_trace_enabled:
            self.start_tracing()

    @beartype
    def start_tracing(self) -> None:
        """Start browser tracing"""
        if self.save_trace_enabled and not self.tracing_started:
            self.context.tracing.start()
            self.tracing_started = True

    @beartype
    def get_page_client(self, page: Page) -> CDPSession:
        """Get the CDP session for the page"""
        return page.context.new_cdp_session(page)

    def _numpy_to_base64(self, image_array: np.ndarray) -> str:
        """Convert numpy array to base64 string for ContentItem"""
        # Convert numpy array to PIL Image
        from PIL import Image
        image = Image.fromarray(image_array)
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Validate the base64 string
        try:
            # Test decode to ensure it's valid
            base64.b64decode(base64_string)
        except Exception as e:
            print(f"Warning: Generated invalid base64 string: {e}")
            print(f"Base64 string length: {len(base64_string)}")
            print(f"Base64 string ends with: {base64_string[-10:]}")
        
        # Return just the base64 string
        return base64_string

    @beartype
    def _get_obs(self) -> dict[str, Observation]:
        """Get the current observation (image screenshot + empty text)"""
        client = self.get_page_client(self.page)
        try:
            image_obs = self.image_processor.process(self.page, client)
            text_obs = self.text_processor.process(self.page, client)
        except Exception as e:
            image_obs = self.image_processor.process(self.page, client)
            text_obs = self.text_processor.process(self.page, client)
        
        # Convert image to base64 for ContentItem compatibility
        image_base64 = self._numpy_to_base64(image_obs)
        
        return {
            "image": image_base64,
            "text": text_obs  # Empty text for now, will be filled with self-reflection, history, etc. later
        }

    @beartype
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, str] | None = None,
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        """
        Reset the environment.
        :param options: options for the environment. The current supported options are:
            - "storage_state": the storage state of the browser. It is a file path to a json file.
        """
        super().reset(seed=seed, options=options)
        
        # Close existing context if it exists and reset_finished is True
        if self.reset_finished:
            try:
                self.context_manager.__exit__(None, None, None)
            except Exception:
                pass  # Ignore errors if context is already closed
        
        # Recreate context and page
        self.context_manager = self.browser.new_context(
            viewport=self.viewport_size,
            record_video_dir=None,
        )
        self.context = self.context_manager.__enter__()
        self.page: Page = self.context.new_page()
        self.tracing_started = False  # Reset tracing state

        if options is not None and "config_file" in options:
            config_file = Path(options["config_file"])
            if config_file.exists():
                self.setup(config_file=config_file)
            else:
                raise ValueError(f"Config file {config_file} does not exist.")
        else:
            self.setup()
        self.reset_finished = True

        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        observation = self._get_obs()
        content = self.page_parser.call(self.page, reasoning="Extracting page content")
        print('*'*50, 'CONTENT', '*'*50)
        is_blocked = self.check_if_blocked(content)

        info = {
            "page": DetachedPage(self.page.url, ""),
            "fail_error": "",
            "observation_metadata": observation,
            "is_blocked": is_blocked
        }

        return observation, info

    @beartype
    def save_trace(self, trace_path: str | Path) -> None:
        """Save browser trace if enabled"""
        if self.save_trace_enabled and self.tracing_started:
            self.context.tracing.stop(path=trace_path)
            self.tracing_started = False

    @beartype
    def close(self) -> None:
        """Close the browser environment"""
        if self.reset_finished:
            self.context_manager.__exit__(None, None, None)

    def step(
        self, action: Action, observation: Any = None
    ) -> tuple[dict[str, Observation], float, bool, bool, dict[str, Any], str]:
        """Execute an action and return the new observation"""
        if not self.reset_finished:
            raise RuntimeError("Call reset first before calling step.")

        success = False
        fail_error = ""

        # Execute pixel action (for grounding model-based actions)
        self.page = execute_pixel_action(
            action, self.page, self.image_processor, observation, self.args
        )
            
        # Wait for page to load
        if self.sleep_after_execution > 0:
            time.sleep(self.sleep_after_execution)

        # Clear interaction point after action execution
        self.image_processor.clear_interaction_point()

        # Get new observation
        observation = self._get_obs()

        # Determine if episode is done (you can customize this logic)
        done = False
        truncated = False

        # Safely get page content with retry mechanism
        page_content = ""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                page_content = self.page.content()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait for navigation to complete
                    continue
                else:
                    # If all retries failed, use empty content
                    page_content = ""
                    print(f"Warning: Could not retrieve page content after {max_retries} attempts: {e}")
                    break

        info = {
            "page": DetachedPage(self.page.url, page_content),
            "fail_error": fail_error,
        }
        
        # Get current URL
        current_url = self.page.url

        return observation, 0.0, done, truncated, info, current_url
    
    def check_if_blocked(self, content: str) -> bool:
        """
        Check if the page content indicates blocking
        
        Args:
            content: Page content to analyze
            
        Returns:
            True if blocked, False otherwise
        """
        if not content:
            return True
            
        # Common blocking indicators
        blocking_indicators = [
            "access denied",
            "blocked",
            "forbidden",
            "403 forbidden",
            "captcha",
        ]
        
        content_lower = content.lower()
        for indicator in blocking_indicators:
            if indicator in content_lower:
                print(f"Blocked by {indicator}")
                return True
        
        # Check for very short content (likely blocking page)
        if len(content.strip()) < 100:
            print("Blocked by short content")
            return True
            
        return False