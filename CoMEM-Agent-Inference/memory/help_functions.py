import torch
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import heapq
from typing import List, Tuple
import base64
from io import BytesIO

class CLIPTextSimilarity:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name,
                    cache_dir='/.cache/huggingface/hub',
                    local_files_only=False).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name,
                    cache_dir='/.cache/huggingface/hub',
                    local_files_only=False)
        
    def get_text_embeddings(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy()
    
    def get_top_similar_questions(self, current_question: str, question_list: List[str], top_n: int = 3) -> List[Tuple[str, float]]:
        """Find the top_n most similar questions to the current question."""
        if not question_list:
            return []
            
        # Get embeddings
        current_embedding = self.get_text_embeddings([current_question])
        all_embeddings = self.get_text_embeddings(question_list)
        
        # Normalize embeddings for cosine similarity
        current_embedding = current_embedding / np.linalg.norm(current_embedding, axis=1, keepdims=True)
        all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(current_embedding, all_embeddings.T)[0]
        
        # Get top N similar questions
        top_indices = heapq.nlargest(top_n+1, range(len(similarities)), key=lambda i: similarities[i])
        
        # Return the questions and their similarity scores
        return [(i, similarities[i]) for i in top_indices if similarities[i] != 1][:top_n]


class CLIPMultimodalSimilarity:
    """CLIP-based multimodal similarity for text and image matching."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
    
    def get_text_embeddings(self, texts):
        """Get text embeddings."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_text_features(**inputs)
        return embeddings.cpu().numpy()
    
    def get_image_embeddings(self, images):
        """Get image embeddings from PIL Images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model.get_image_features(**inputs)
        return embeddings.cpu().numpy()
    
    def get_base64_image_embeddings(self, base64_images):
        """Get image embeddings from base64 encoded images."""
        images = []
        for base64_img in base64_images:
            try:
                if isinstance(base64_img, dict) and 'url' in base64_img:
                    base64_img = base64_img['url']
                # Remove data URL prefix if present
                if base64_img.startswith('data:image'):
                    base64_img = base64_img.split(',')[1]
                
                # Decode base64 to image
                image_bytes = base64.b64decode(base64_img)
                image = Image.open(BytesIO(image_bytes))
                images.append(image)
            except Exception as e:
                print(f"Error processing base64 image: {e}")
                # Return None for failed images
                return None
        
        if images:
            return self.get_image_embeddings(images)
        return None
    
    def get_multimodal_embeddings(self, texts, base64_images=None):
        """Get combined text and image embeddings."""
        text_embeddings = self.get_text_embeddings(texts)
        
        if base64_images is not None:
            image_embeddings = self.get_base64_image_embeddings(base64_images)
            if image_embeddings is not None:
                # Combine text and image embeddings (simple concatenation)
                # You could also use weighted combination or other fusion methods
                combined_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
                return combined_embeddings
            else:
                # If image processing failed, create zero embeddings for images to maintain dimension consistency
                zero_image_embeddings = np.zeros_like(text_embeddings)
                combined_embeddings = np.concatenate([text_embeddings, zero_image_embeddings], axis=1)
                return combined_embeddings
        
        # If no images provided, return text embeddings only
        return text_embeddings
    
    def calculate_similarity(self, query_embedding, candidate_embeddings):
        """Calculate cosine similarity between query and candidates."""
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        candidates_norm = candidate_embeddings / np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.dot(query_norm, candidates_norm.T)[0]
        return similarities


def main():
    # Example usage
    clip_similarity = CLIPTextSimilarity()

    questions = [
        "How do I install Python on Windows?",
        "What's the best way to learn Python?",
        "How can I set up a virtual environment in Python?",
        "What are the key features of Python 3.9?",
        "How do I debug Python code efficiently?",
        "What's the syntax for list comprehension in Python?",
        "How to install Python packages using pip?",
        "What are Python decorators?",
        "How to handle exceptions in Python?",
        "What is the difference between Python 2 and Python 3?"
    ]

    current_question = "What's the easiest way to install Python on my Windows computer?"
    top_similar = clip_similarity.get_top_similar_questions(current_question, questions)

    # for question, score in top_similar:
    #     print(f"Score: {score:.4f} - {question}")

if __name__ == "__main__":
    main()