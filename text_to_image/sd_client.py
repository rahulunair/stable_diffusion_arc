import requests
import random

model_ids = [
    "CompVis/stable-diffusion-v1-4",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-xl-base-1.0",
]

print("Available models are:")
for i, model_id in enumerate(model_ids):
    print(f"{i + 1}. {model_id}")

selected_model_index = (
    int(
        input(
            "Select a model by entering its number (or press enter to use the default model): "
        )
    )
    - 1
)
model_id = (
    model_ids[selected_model_index]
    if 0 <= selected_model_index < len(model_ids)
    else model_ids[0]
)
prompt = input("Please enter your prompt: ")
num_images = 0
try:
    num_images = int(input("How many images have to be generated: "))
except Exception:
    num_images = 0
if num_images <= 0:
    num_images = 1

enhancements = [
    "dark",
    "purple light",
    "dreaming",
    "cyberpunk",
    "ancient" ", rustic",
    "gothic",
    "historical",
    "punchy",
    "photo" "vivid colors",
    "4k",
    "bright",
    "exquisite",
    "painting",
    "art",
    "fantasy [,/organic]",
    "detailed",
    "trending in artstation fantasy",
    "electric",
    "night",
]

prompt = prompt + " " + " ".join(random.sample(enhancements, 5))
print(f"Using enhanced prompt: {prompt}")

response = requests.post(
    "http://localhost:5000/generate_images",
    json={
        "model_id": model_id,
        "prompt": prompt,
        "num_images": num_images
    }
)

data = response.json()
if data["status"] == "success":
    print("Image generation successful. Images saved at: ", ", ".join(data["image_paths"]))
else:
    print('Image generation failed. Error:{data["message"]}')

