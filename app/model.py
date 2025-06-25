# from model import load_model, generate_image
# from app.model import load_model, generate_image
from diffusers import StableDiffusionPipeline
import torch

from PIL import Image

def load_model():
    """
    Loads the Stable Diffusion model.
    You need to be logged in to Hugging Face CLI or use a token.
    """
    model_id = "runwayml/stable-diffusion-v1-5"  # Or another compatible model

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        revision="fp16" if torch.cuda.is_available() else None,
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    return pipe

def generate_image(pipe, prompt):
    """
    Generates an image from the given text prompt.
    """
    if not prompt:
        return None

    result = pipe(prompt)
    image = result.images[0] if result and result.images else None

    return image
