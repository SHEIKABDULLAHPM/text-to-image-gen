import streamlit as st
from model import load_model, generate_image
import random
import io

# ------------------- Setup -------------------
st.set_page_config(page_title="GenAI Image Generator", layout="centered")

# ------------------- Sidebar Controls -------------------
st.sidebar.title("⚙️ Generator Settings")
use_smart_prompt = st.sidebar.checkbox("🧠 Smart Prompt Enhancer", value=True)
num_images = st.sidebar.slider("Number of Images", 1, 4, 1)
selected_resolution = st.sidebar.selectbox("Image Resolution", ["512x512", "768x768", "1024x1024"])
negative_prompt = st.sidebar.text_input("Negative Prompt (optional)", placeholder="e.g. blurry, text, watermark")
refine = st.sidebar.checkbox("Use SDXL Refiner", value=True)

# Parse resolution
width, height = map(int, selected_resolution.split("x"))

# ------------------- Load Model -------------------
@st.cache_resource
def get_pipe():
    return load_model()

pipe = get_pipe()

# ------------------- Prompt Examples -------------------
prompt_examples = [
    "a futuristic cyberpunk city at night",
    "a magical forest with glowing mushrooms",
    "a cute baby elephant playing with water",
    "a sci-fi space station orbiting Earth",
    "a serene beach at sunset with palm trees"
]

# ------------------- Main UI -------------------
st.title("🖼️ AI Image Generator")
st.markdown("Generate stunning images from simple text prompts using GenAI diffusion models.")

# Prompt input
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.text_input("Enter your prompt here:", placeholder="e.g. a dragon flying over snowy mountains")
with col2:
    if st.button("🎲 Surprise Me"):
        prompt = random.choice(prompt_examples)
        st.experimental_rerun()

# ------------------- Generate Button -------------------
if st.button("🚀 Generate"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image(s)... Please wait ⏳"):
            images = generate_image(
                pipe,
                prompt,
                negative_prompt=negative_prompt,
                num_images=num_images,
                height=height,
                width=width,
                refine=refine,
                smart_prompt=use_smart_prompt
            )

        if images:
            st.subheader("🖼️ Generated Image(s)")
            cols = st.columns(num_images)
            for i in range(num_images):
                with cols[i]:
                    st.image(images[i], use_container_width=True)

                    # Download Button
                    buf = io.BytesIO()
                    images[i].save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.download_button(
                        label="📥 Download",
                        data=byte_im,
                        file_name=f"generated_image_{i+1}.png",
                        mime="image/png"
                    )
        else:
            st.error("❌ Failed to generate image.")
