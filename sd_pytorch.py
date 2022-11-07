# coding: utf-8
import random

import intel_extension_for_pytorch
import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

keywords = [
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
    "night"
]
# https://www.poetryfoundation.org/poems/46565/ozymandias
prompts = [
    "I met a traveller from an antique land",
    "Who said—“Two vast and trunkless legs of stone",
    "Stand in the desert. . . . Near them, on the sand",
    "Half sunk a shattered visage lies, whose frown",
    "And wrinkled lip, and sneer of cold command",
    "Tell that its sculptor well those passions read",
    "Which yet survive, stamped on these lifeless things",
    "The hand that mocked them, and the heart that fed",
    "And on the pedestal, these words appear",
    "My name is Ozymandias, King of Kings",
    "Look on my Works, ye Mighty, and despair!",
    "Nothing beside remains. Round the decay",
    "Of that colossal Wreck, boundless and bare",
    "The lone and level sands stretch far away",
]

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="<your_token>",
)

pipe = pipe.to("xpu")
for prompt in prompts:
    fname = prompt[:5]
    prompt = prompt + " " + " ".join(random.sample(keywords, 5))
    print(f"prompt used: {prompt}")
    image = pipe(prompt).images[0]
    image.save(f"{fname}.png")
