from diffusers import StableDiffusionPipeline
import torch
from config.authtoken import HUGGINGFACE_TOKEN

# Function to generate image
def generate_image(prompt):
    # Initialize the pipeline with the Hugging Face token
    pipe = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5", 
        use_auth_token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float32
    )
    pipe = pipe.to("cpu")  # Use CPU since no GPU is available
    
    # Generate image from prompt
    image = pipe(prompt).images[0]

    # Save the generated image
    image_path = "generated_img.png"
    image.save(image_path)
    print(f"Image saved as {image_path}")
    return image_path

# Main function to run the app
def main():
    # Define the prompt (change as needed)
    prompt = "a photo of a cat wearing hoodie"
    
    # Generate and save the image
    image_path = generate_image(prompt)
    print(f"Generated image: {image_path}")

if __name__ == "__main__":
    main()