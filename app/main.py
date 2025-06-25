import streamlit as st
from model import load_model, generate_image

import io

st.set_page_config(page_title="GenAI Image Generator", layout="centered")

# App Title
st.title("🖼️ AI Image Generator")
st.markdown("Generate stunning images from simple text prompts using GenAI diffusion models.")

# Load the model (only once)
@st.cache_resource
def get_pipe():
    return load_model()

pipe = get_pipe()

# Prompt input
prompt = st.text_input("Enter a prompt (e.g. 'a fantasy castle in the clouds'):")

# Generate button
if st.button("Generate Image") and prompt:
    with st.spinner("Generating... please wait ⏳"):
        image = generate_image(pipe, prompt)

        if image:
            st.image(image, caption="Generated Image", use_column_width=True)

            # Download option
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(label="📥 Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")
        else:
            st.error("Failed to generate image. Try again.")
