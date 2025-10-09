import base64
import io
from PIL import Image

def clean_url(url: str) -> str:
    url = str(url)
    if url.endswith("/"):
        url = url[:-1]
    return url

def clean_answer(answer: str) -> str:
    answer = answer.strip("'").strip('"')
    answer = answer.lower()
    return answer

def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')