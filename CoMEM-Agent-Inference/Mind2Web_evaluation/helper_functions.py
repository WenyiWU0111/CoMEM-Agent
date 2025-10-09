import base64
import io

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

def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0  
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0      
    else:
        raise ValueError(f"Unknown mode: {mode}")