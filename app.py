from diffusers import StableDiffusionPipeline
import torch
import gradio as gr
from config.authtoken import HUGGINGFACE_TOKEN

# Initialize the pipeline once
pipe = StableDiffusionPipeline.from_pretrained(
    "sd-legacy/stable-diffusion-v1-5",
    use_auth_token=HUGGINGFACE_TOKEN,
    torch_dtype=torch.float32
)
pipe = pipe.to("cpu")  # Use CPU since no GPU is available

# Function to generate image
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Gradio interface with custom CSS
def gradio_app():
    custom_css = """
    .title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #4CAF50;
    }
    .description {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #555;
    }
    """
    gr.Interface(
        fn=generate_image,
        inputs=gr.Textbox(label="Enter your prompt"),
        outputs=gr.Image(type="pil", label="Generated Image"),
        title="<div class='title'>Neural Artworks</div>",
        description="<div class='description'>"
                    "Welcome to <strong>Neural Artworks</strong>! 🎨<br>"
                    "Generate stunning images from text prompts. "
                    "Describe the image you want, and let the AI bring it to life!"
                    "</div>",
        css=custom_css
    ).launch()

if __name__ == "__main__":
    gradio_app()
