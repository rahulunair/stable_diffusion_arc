# coding: utf-8
from multiprocessing import Process
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
    "night",
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

pipe1 = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="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",
)

pipe2 = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="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",
)

pipe3 = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="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",
)
pipe4 = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="fp16",
    use_auth_token="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",
)
def infer(pipe):
    for _ in range(2):
        prompt = "I met a traveller from an antique land, dreaming, 4k, vivid colors, hi-res"
        print(f"prompt used: {prompt}")
        image = pipe(prompt).images[0]
        image.save("./test.png")

pipe1 = pipe1.to("xpu:1")
pipe2 = pipe2.to("xpu:2")
pipe3 = pipe3.to("xpu:3")
pipe4 = pipe4.to("xpu:4")

p1 = Process(target=infer, args=(pipe1,))
p2 = Process(target=infer, args=(pipe2,))
p3 = Process(target=infer, args=(pipe4,))
p4 = Process(target=infer, args=(pipe4,))
p1.start(); p2.start();p3.start();p4.start()
p1.join(); p2.join(); p3.join(); p4.join()
