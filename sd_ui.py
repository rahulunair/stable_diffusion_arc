import random

import gradio as gr
import requests

from PIL import Image


def concatenate_images(image_list, cols=2):
    """
    Concatenates images into a grid.

    Args:
    - image_list: List of PIL Image objects
    - cols: Number of columns in the grid

    Returns:
    - PIL Image object representing the grid of images
    """
    rows = len(image_list) // cols + (len(image_list) % cols > 0)
    max_width = max(image.width for image in image_list)
    max_height = max(image.height for image in image_list)
    grid = Image.new("RGB", (max_width * cols, max_height * rows))
    for i, image in enumerate(image_list):
        grid.paste(image, (i % cols * max_width, i // cols * max_height))
    return grid


def generate_images(model_name: int, prompt: str, num_images: int):
    """
    Generates images based on the model, prompt, and number of images chosen by the user.

    Args:
        model_name (int): The name of the model chosen by the user.
        prompt (str): The prompt given by the user.
        num_images (int): The number of images to generate.

    Returns:
        PIL Image: A grid of generated images.
    """
    model_ids = [
        "CompVis/stable-diffusion-v1-4",
        "stabilityai/stable-diffusion-2-1",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]

    enhancements = [
        "dark",
        "purple light",
        "dreaming",
        "cyberpunk",
        "ancient rustic",
        "gothic",
        "historical",
        "punchy",
        "photo vivid colors",
        "4k",
        "bright",
        "exquisite",
        "painting",
        "art",
        "fantasy organic",
        "detailed",
        "trending in artstation fantasy",
        "electric",
        "night",
    ]

    prompt = prompt + " " + " ".join(random.sample(enhancements, 5))

    response = requests.post(
        "http://localhost:5000/generate_images",
        json={
            "model_id": model_name,
            "prompt": prompt,
            "num_images": num_images,
        },
    )

    data = response.json()
    if data["status"] == "success":
        images = [Image.open(path) for path in data["image_paths"]]
        image_grid = concatenate_images(images)
        return image_grid
    else:
        raise Exception(f'Image generation failed. Error: {data["message"]}')


iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.inputs.Dropdown(
            [
                "CompVis/stable-diffusion-v1-4",
                "stabilityai/stable-diffusion-2-1",
                "stabilityai/stable-diffusion-xl-base-1.0",
            ],
            label="Model",
        ),
        gr.inputs.Textbox(lines=1, label="Prompt"),
        gr.inputs.Slider(
            minimum=1, maximum=10, step=1, default=1, label="Number of Images"
        ),
    ],
    outputs=[gr.outputs.Image(type="pil", label="Generated Images")],
    title="Stable Diffusion on Intel XPUs",
    description="Generate images using the Stable Diffusion models on Intel XPUs.",
    theme="huggingface",
)

iface.launch()
