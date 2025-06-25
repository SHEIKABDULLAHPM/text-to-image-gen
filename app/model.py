from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from transformers import pipeline

# ------------------- Smart Prompt Enhancer -------------------

# Load once: zero-shot classifier model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def refine_prompt(prompt):
    """
    Enhances abstract prompts with visual styling based on inferred tone/context.
    """
    candidate_labels = ["dreamy", "futuristic", "realistic", "abstract", "surreal", "emotional", "fantasy"]
    result = classifier(prompt, candidate_labels)
    top_label = result["labels"][0]

    # Visual style enhancements based on prompt classification
    visual_modifiers = {
        "dreamy": "ethereal, soft lighting, hazy atmosphere",
        "futuristic": "neon lights, sci-fi cityscape, high-tech",
        "realistic": "photo-realistic, natural lighting, DSLR clarity",
        "abstract": "geometric shapes, vivid colors, artistic brush strokes",
        "surreal": "melting forms, bizarre landscapes, dream logic",
        "emotional": "expressive characters, bold colors, dynamic angles",
        "fantasy": "magical creatures, glowing elements, fantasy setting"
    }

    enriched_prompt = f"{prompt}, {visual_modifiers.get(top_label, '')}"
    return enriched_prompt

# ------------------- Load Base SDXL Model -------------------

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None
    )
    pipe = pipe.to(device)

    if device == "cuda" and hasattr(torch, "compile"):
        pipe.unet = torch.compile(pipe.unet)

    return pipe

# ------------------- Optional Refiner -------------------

def load_refiner(pipe):
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16
    ).to(pipe.device)

    return refiner

# ------------------- Image Generation -------------------

def generate_image(pipe, prompt, negative_prompt="", num_images=1, height=1024, width=1024, refine=True, smart_prompt=True):
    if not prompt:
        return []

    # Optional: Enhance the prompt with style/context
    if smart_prompt:
        prompt = refine_prompt(prompt)

    prompts = [prompt] * num_images
    negative_prompts = [negative_prompt] * num_images

    result = pipe(
        prompt=prompts,
        negative_prompt=negative_prompts,
        height=height,
        width=width,
        num_inference_steps=25,
        guidance_scale=7.5
    )

    images = result.images

    # Optional: SDXL Refiner
    if refine:
        refiner = load_refiner(pipe)
        images = refiner(
            prompt=prompts,
            image=images,
            strength=0.3
        ).images

    return images
