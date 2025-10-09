"""Simplified observation processors for image and text observations"""
import numpy as np
import numpy.typing as npt
from beartype import beartype
from PIL import Image, ImageDraw
from playwright.sync_api import CDPSession, Page
from typing import TypedDict
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFont
from io import StringIO

from .utils import Observation, png_bytes_to_numpy, BrowserConfig, BrowserInfo


class ObservationMetadata(TypedDict):
    """Simple observation metadata for compatibility with the original MMInA structure"""
    pass


class ObservationProcessor:
    """Base class for observation processors"""
    def process(self, page: Page, client: CDPSession) -> Observation:
        raise NotImplementedError


class SimpleTextObservationProcessor(ObservationProcessor):
    """Simple text observation processor that returns empty text for now"""
    
    def __init__(self, args=None):
        self.args = args

    @beartype
    def process(self, page: Page, client: CDPSession) -> str:
        """Process the page and return empty text (placeholder for future text processing)"""
        # TODO: Add self-reflection, history description, etc. here later
        return ""


class SimpleImageObservationProcessor(ObservationProcessor):
    """Simple image observation processor that only handles screenshots"""
    
    def __init__(self, args=None):
        self.args = args
        self.interaction_coords = None  # Store the current interaction coordinates
        self.viewport_size = {"width": 1280, "height": 720}

    def set_interaction_point(self, x: float, y: float):
        """Set the interaction point coordinates for drawing a red circle"""
        self.interaction_coords = (x, y)

    def set_interaction_point_from_action(self, action):
        """Set interaction point from action coordinates"""
        if action and 'action_inputs' in action:
            start_box = action['action_inputs'].get('start_box')
            if start_box:
                try:
                    if isinstance(start_box, str):
                        coords = eval(start_box)
                    else:
                        coords = start_box
                    
                    if len(coords) >= 2:
                        if len(coords) == 4:
                            # If it's a bounding box, calculate center
                            x1, y1, x2, y2 = coords
                            x = (x1 + x2) / 2
                            y = (y1 + y2) / 2
                        else:
                            # If it's already coordinates
                            x, y = coords[0], coords[1]
                        
                        self.set_interaction_point(x, y)
                        return True
                except Exception as e:
                    print(f"Error parsing action coordinates: {e}")
        return False

    def clear_interaction_point(self):
        """Clear the interaction point"""
        self.interaction_coords = None

    def get_page_bboxes(self, page: Page) -> list[list[float]]:
        """JavaScript code to return bounding boxes and other metadata from HTML elements."""
        js_script = """
        (() => {
            const interactableSelectors = [
                'a[href]:not(:has(img))', 'a[href] img', 'button', 'input:not([type="hidden"])', 'textarea', 'select',
                '[tabindex]:not([tabindex="-1"])', '[contenteditable="true"]', '[role="button"]', '[role="link"]',
                '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]', '[draggable="true"]',
                '.btn', 'a[href="/notifications"]', 'a[href="/submit"]', '.fa.fa-star.is-rating-item', 'input[type="checkbox"]'

            ];

            const textSelectors = ['p', 'span', 'div:not(:has(*))', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article'];
            const modifiedTextSelectors = textSelectors.map(selector =>
                `:not(${interactableSelectors.join(', ')}):not(style) > ${selector}`
            );

            const combinedSelectors = [...interactableSelectors, ...modifiedTextSelectors];
            const elements = document.querySelectorAll(combinedSelectors.join(', '));

            const pixelRatio = window.devicePixelRatio;
            let csvContent = "ID,Element,Top,Right,Bottom,Left,Width,Height,Alt,Class,Id,TextContent,Interactable\\n";
            let counter = 1;

            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) return;
                let altText = element.getAttribute('alt') || '';
                altText = altText.replace(/"/g, ''); // Escape double quotes in alt text
                const classList = element.className || '';
                const id = element.id || '';
                let textContent = element.textContent || '';
                textContent = textContent.replace(/"/g, ''); // Escape double quotes in textContent

                // Determine if the element is interactable
                const isInteractable = interactableSelectors.some(selector => element.matches(selector));

                const dataString = [
                    counter, element.tagName, (rect.top + window.scrollY) * pixelRatio,
                    (rect.right + window.scrollX) * pixelRatio, (rect.bottom + window.scrollY) * pixelRatio,
                    (rect.left + window.scrollX) * pixelRatio, rect.width * pixelRatio, rect.height * pixelRatio,
                    altText, classList, id, textContent, isInteractable
                ].map(value => `"${value}"`).join(",");

                csvContent += dataString + "\\n";
                counter++;
            });

            return csvContent;
        })();
        """
        # Save the bbox as a CSV
        csv_content = page.evaluate(js_script)
        return csv_content

    def draw_bounding_boxes(
        self,
        data_string,
        screenshot_img,
        viewport_size=None,
        add_ids=True,
        bbox_color=None,
        min_width=8,
        min_height=8,
        bbox_padding=0,
        bbox_border=2,
        plot_ids=None,
    ):
        """
        min_width and min_height: Minimum dimensions of the bounding box to be plotted.
        """
        # Read CSV data
        df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
        df["Area"] = df["Width"] * df["Height"]
        # Remove bounding boxes that are clipped.
        b_x, b_y = (
            self.browser_config["win_left_bound"],
            self.browser_config["win_upper_bound"],
        )
        if viewport_size is not None:
            df = df[
                (df["Bottom"] - b_y >= 0)
                & (df["Top"] - b_y <= viewport_size["height"])
                & (df["Right"] - b_x >= 0)
                & (df["Left"] - b_x <= viewport_size["width"])
            ]
            viewport_area = viewport_size["width"] * viewport_size["height"]
            # Filter out bounding boxes that too large (more than 80% of the viewport)
            df = df[df["Area"] <= 0.8 * viewport_area]

        # Open the screenshot image
        img = screenshot_img.copy()
        draw = ImageDraw.Draw(img)

        # Load a TTF font with a larger size
        font_path = "media/SourceCodePro-SemiBold.ttf"
        font_size, padding = 16, 2
        font = ImageFont.truetype(font_path, font_size)

        # Create a color cycle using one of the categorical color palettes in matplotlib
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bbox_id2visid = {}
        bbox_id2desc = {}
        index = 0
        id2center = {}
        existing_text_rectangles = []
        text_to_draw = []
        # Provide [id] textContent inputs to the model as text.
        text_content_elements = []
        text_content_text = set()  # Store text of interactable elements

        # Iterate through each row in the CSV and draw bounding boxes
        for _, row in df.iterrows():
            if not row["Interactable"]:
                content = ""
                # Add image alt-text to the text representation.
                if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                    content += row["Alt"]
                # Add HTML textContent (if any) to the text representation.
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"].strip().replace("\n", "").replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters to avoid having too much text

                # Check if the text is a CSS selector
                if content and not (content.startswith(".") and "{" in content):
                    # Add elements which are not interactable as StaticText
                    if content not in text_content_text:
                        text_content_elements.append(f"[] [StaticText] [{content}]")
                        text_content_text.add(content)
                continue

            if (plot_ids is not None) and (row["ID"] not in plot_ids):
                continue

            unique_id = str(index + 1)
            bbox_id2visid[row["ID"]] = (
                unique_id  # map the bounding box ID to the unique character ID
            )
            top, right, bottom, left, width, height = (
                row["Top"],
                row["Right"],
                row["Bottom"],
                row["Left"],
                row["Width"],
                row["Height"],
            )
            left, right, top, bottom = left - b_x, right - b_x, top - b_y, bottom - b_y
            id2center[unique_id] = (
                (left + right) / 2,
                (bottom + top) / 2,
                width,
                height,
            )

            if width >= min_width and height >= min_height:
                # Get the next color in the cycle
                color = bbox_color or color_cycle[index % len(color_cycle)]
                draw.rectangle(
                    [
                        left - bbox_padding,
                        top - bbox_padding,
                        right + bbox_padding,
                        bottom + bbox_padding,
                    ],
                    outline=color,
                    width=bbox_border,
                )
                bbox_id2desc[row["ID"]] = color

                # Draw the text on top of the rectangle
                if add_ids:
                    # Calculate list of possible text positions
                    text_positions = [
                        (left - font_size, top - font_size),  # Top-left corner
                        (
                            left,
                            top - font_size,
                        ),  # A little to the right of the top-left corner
                        (right, top - font_size),  # Top-right corner
                        (
                            right - font_size - 2 * padding,
                            top - font_size,
                        ),  # A little to the left of the top-right corner
                        (left - font_size, bottom),  # Bottom-left corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                        (
                            left,
                            bottom,
                        ),  # A little to the right of the bottom-left corner
                        (
                            right - font_size - 2 * padding,
                            bottom,
                        ),  # A little to the left of the bottom-right corner
                    ]
                    text_width = draw.textlength(unique_id, font=font)
                    text_height = font_size  # Assume the text is one line

                    if viewport_size is not None:
                        for text_position in text_positions:
                            new_text_rectangle = [
                                text_position[0] - padding,
                                text_position[1] - padding,
                                text_position[0] + text_width + padding,
                                text_position[1] + text_height + padding,
                            ]

                            # Check if the new text rectangle is within the viewport
                            if (
                                new_text_rectangle[0] >= 0
                                and new_text_rectangle[1] >= 0
                                and new_text_rectangle[2] <= viewport_size["width"]
                                and new_text_rectangle[3] <= viewport_size["height"]
                            ):
                                # If the rectangle is within the viewport, check for overlaps
                                overlaps = False
                                for existing_rectangle in existing_text_rectangles:
                                    if self.rectangles_overlap(
                                        new_text_rectangle,
                                        existing_rectangle,
                                        padding * 2,
                                    ):
                                        overlaps = True
                                        break

                                if not overlaps:
                                    break
                            else:
                                # If the rectangle is outside the viewport, try the next position
                                continue
                    else:
                        # If none of the corners work, move the text rectangle by a fixed amount
                        text_position = (
                            text_positions[0][0] + padding,
                            text_positions[0][1],
                        )
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                    existing_text_rectangles.append(new_text_rectangle)
                    text_to_draw.append(
                        (new_text_rectangle, text_position, unique_id, color)
                    )

                    content = ""
                    if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                        content += row["Alt"]
                    if pd.notna(row["TextContent"]):
                        content += (
                            row["TextContent"]
                            .strip()
                            .replace("\n", "")
                            .replace("\t", "")
                        )[
                            :200
                        ]  # Limit to 200 characters
                    text_content_elements.append(
                        f"[{unique_id}] [{row['Element']}] [{content}]"
                    )
                    if content in text_content_text:
                        # Remove text_content_elements with content
                        text_content_elements = [
                            element
                            for element in text_content_elements
                            if element.strip() != content
                        ]
                    text_content_text.add(content)

            index += 1

        for text_rectangle, text_position, unique_id, color in text_to_draw:
            # Draw a background rectangle for the text
            draw.rectangle(text_rectangle, fill=color)
            draw.text(text_position, unique_id, font=font, fill="white")

        content_str = "\n".join(text_content_elements)
        return img, id2center, content_str

    def rectangles_overlap(self, rect1, rect2, padding):
        """
        Check if two rectangles overlap.
        Each rectangle is represented as a list [x1, y1, x2, y2].
        """
        return not (
            rect1[2] < rect2[0] + padding
            or rect1[0] > rect2[2] - padding
            or rect1[1] > rect2[3] - padding
            or rect1[3] < rect2[1] + padding
        )

    def process(self, page: Page) -> npt.NDArray[np.uint8]:
        try:
            browser_info = self.fetch_browser_info(page)
        except Exception:
            page.wait_for_load_state("load", timeout=500)
            browser_info = self.fetch_browser_info(page)

        self.browser_config = browser_info["config"]

        if self.observation_type == "image_som":
            # Produce the SoM image, with bounding boxes
            try:
                screenshot_bytes = page.screenshot()
                som_bboxes = self.get_page_bboxes(page)
                screenshot_img = Image.open(BytesIO(screenshot_bytes))
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
            except:
                try:
                    page.wait_for_event("load")
                    screenshot_bytes = page.screenshot()
                    screenshot_img = Image.open(BytesIO(screenshot_bytes))
                except:
                    screenshot_img = Image.new('RGB', (1280, 720), color='white')
                som_bboxes = self.get_page_bboxes(page)
                bbox_img, id2center, content_str = self.draw_bounding_boxes(
                    som_bboxes,
                    screenshot_img,
                    viewport_size=self.viewport_size,
                )
                self.som_id_info = id2center
                self.meta_data["obs_nodes_info"] = id2center
                screenshot_som = np.array(bbox_img)
                return screenshot_som, content_str
        else:
            try:
                screenshot = png_bytes_to_numpy(page.screenshot())
            except:
                try:
                    page.wait_for_event("load")
                    screenshot = png_bytes_to_numpy(page.screenshot())
                except:
                    screenshot = np.zeros((720, 1280, 3), dtype=np.uint8)
                return screenshot, ""
            return screenshot, ""
        
    def fetch_browser_info(self, page: Page):
        client = page.context.new_cdp_session(page)
        # extract domtree
        tree = client.send(
            "DOMSnapshot.captureSnapshot",
            {
                "computedStyles": [],
                "includeDOMRects": True,
                "includePaintOrder": True,
            },
        )
        client.detach()
        # calibrate the bounds, in some cases, the bounds are scaled somehow
        bounds = tree["documents"][0]["layout"]["bounds"]
        b = bounds[0]
        n = b[2] / self.viewport_size["width"]
        bounds = [[x / n for x in bound] for bound in bounds]
        tree["documents"][0]["layout"]["bounds"] = bounds
        # add union bound placeholder
        tree["documents"][0]["layout"]["unionBounds"] = [None for _ in bounds]

        # extract browser info
        win_upper_bound = page.evaluate("window.pageYOffset")
        win_left_bound = page.evaluate("window.pageXOffset")
        win_width = page.evaluate("window.screen.width")
        win_height = page.evaluate("window.screen.height")
        win_right_bound = win_left_bound + win_width
        win_lower_bound = win_upper_bound + win_height
        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        assert device_pixel_ratio == 1.0, "devicePixelRatio is not 1.0"

        config: BrowserConfig = {
            "win_upper_bound": win_upper_bound,
            "win_left_bound": win_left_bound,
            "win_width": win_width,
            "win_height": win_height,
            "win_right_bound": win_right_bound,
            "win_lower_bound": win_lower_bound,
            "device_pixel_ratio": device_pixel_ratio,
        }

        # assert len(tree['documents']) == 1, "More than one document in the DOM tree"
        info: BrowserInfo = {"DOMTree": tree, "config": config}

        return info

    def get_element_center(self, element_id: str) -> tuple[float, float]:

        browser_config = self.browser_config
        center_x, center_y, width, height = self.som_id_info[element_id]
        return (
            center_x / self.viewport_size["width"],
            center_y / self.viewport_size["height"],
        )
        
        
    def process(self, page: Page, client: CDPSession):
        """Process the page and return a screenshot as numpy array"""
        try:
            screenshot_bytes = page.screenshot()
        except:
            page.wait_for_timeout(60000)
            try:
                screenshot_bytes = page.screenshot()
            except:
                screenshot_bytes = np.zeros((720, 1280, 3), dtype=np.uint8)
        browser_info = self.fetch_browser_info(page)
        self.browser_config = browser_info["config"]
        
        som_bboxes = self.get_page_bboxes(page)
        screenshot_img = Image.open(BytesIO(screenshot_bytes))
        try:
            bbox_img, id2center, content_str = self.draw_bounding_boxes(
                som_bboxes,
                screenshot_img,
                viewport_size={"width": 1280, "height": 720},
            )
            self.som_id_info = id2center
            screenshot_som = np.array(bbox_img)
        except Exception as e:
            return np.array(screenshot_img)
        return screenshot_som
        