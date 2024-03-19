import os
import warnings

warnings.filterwarnings("ignore")

import random
import requests
import torch
import intel_extension_for_pytorch as ipex
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import DPMSolverMultistepScheduler
import torch.nn as nn
import time
from typing import List, Dict, Tuple
import validators
import numpy as np


class Img2ImgModel:
    """
    This class creates a model for transforming images based on given prompts.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "xpu",
        torch_dtype: torch.dtype = torch.bfloat16,
        optimize: bool = True,
        warmup: bool = True,
        scheduler: bool = True,
    ) -> None:
        """
        Initialize the model with the specified parameters.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            device (str, optional): The device to run the model on. Defaults to "xpu".
            torch_dtype (torch.dtype, optional): The data type to use for the model. Defaults to torch.float16.
            optimize (bool, optional): Whether to optimize the model. Defaults to True.
        """
        self.device = device
        self.data_type = torch_dtype
        self.scheduler = scheduler
        self.generator = torch.Generator()  # .manual_seed(99)
        self.pipeline = self._load_pipeline(model_id_or_path, torch_dtype)
        if optimize:
            start_time = time.time()
            print("Optimizing the model...")
            self.optimize_pipeline()
            print(
                "Optimization completed in {:.2f} seconds.".format(
                    time.time() - start_time
                )
            )
        if warmup:
            self.warmup_model()

    def _load_pipeline(
        self, model_id_or_path: str, torch_dtype: torch.dtype
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Load the pipeline for the model.

        Args:
            model_id_or_path (str): The ID or path of the pre-trained model.
            torch_dtype (torch.dtype): The data type to use for the model.

        Returns:
            StableDiffusionImg2ImgPipeline: The loaded pipeline.
        """
        print("Loading the model...")
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id_or_path, torch_dtype=torch_dtype
        )
        pipeline = pipeline.to(self.device)
        if self.scheduler:
            pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                pipeline.scheduler.config
            )
        print("Model loaded.")
        return pipeline

    def _optimize_pipeline(
        self, pipeline: StableDiffusionImg2ImgPipeline
    ) -> StableDiffusionImg2ImgPipeline:
        """
        Optimize the pipeline of the model.

        Args:
            pipeline (StableDiffusionImg2ImgPipeline): The pipeline to optimize.

        Returns:
            StableDiffusionImg2ImgPipeline: The optimized pipeline.
        """
        
        for attr in dir(pipeline):
            try:
                if isinstance(getattr(pipeline, attr), nn.Module):
                    setattr(
                        pipeline,
                        attr,
                        ipex.optimize(
                            getattr(pipeline, attr).eval(),
                            dtype=pipeline.text_encoder.dtype,
                            inplace=True,
                        ),
                    )
            except AttributeError:
                pass
        return pipeline

    def optimize_pipeline(self) -> None:
        """
        Optimize the pipeline of the model.
        """
        self.pipeline = self._optimize_pipeline(self.pipeline)

    def get_image_from_url(self, url: str, path: str) -> Image.Image:
        """
        Get an image from a URL or from a local path if it exists.

        Args:
            url (str): The URL of the image.
            path (str): The local path of the image.

        Returns:
            Image.Image: The loaded image.
        """
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(
                f"Failed to download image. Status code: {response.status_code}"
            )
        if not response.headers["content-type"].startswith("image"):
            raise Exception(
                f"URL does not point to an image. Content type: {response.headers['content-type']}"
            )
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(path)
        img = img.resize((768, 512))
        return img

    def warmup_model(self):
        """
        Warms up the model by generating a sample image.
        """
        print("Setting up model...")
        start_time = time.time()
        image_url = "https://user-images.githubusercontent.com/786476/256401499-f010e3f8-6f8d-4e9f-9d1f-178d3571e7b9.png"
        try:
            self.generate_images(
                image_url=image_url,
                prompt="A beautiful day",
                num_images=1,
                save_path="/tmp",
            )
        except Exception:
            print("model warmup delayed...")
        print(
            "Model is set up and ready! Warm-up completed in {:.2f} seconds.".format(
                time.time() - start_time
            )
        )

    def get_inputs(self, prompt, batch_size=1):
        self.generator = [torch.Generator() for i in range(batch_size)]
        prompts = batch_size * [prompt]
        return {"prompt": prompts, "generator": self.generator}

    def generate_images(
        self,
        prompt: str,
        image_url: str,
        num_images: int = 5,
        num_inference_steps: int = 30,
        strength: float = 0.75,
        guidance_scale: float = 7.5,
        save_path: str = "output",
        batch_size: int = 1,
    ):
        """
        Generate images based on the provided prompt and variations.

        Args:
            prompt (str): The base prompt for the generation.
            image_url (str): The URL of the seed image.
            variations (List[str]): The list of variations to apply to the prompt.
            num_images (int, optional): The number of images to generate. Defaults to 5.
            num_inference_steps (int, optional): Number of noise removal steps.
            strength (float, optional): The strength of the transformation. Defaults to 0.75.
            guidance_scale (float, optional): The scale of the guidance. Defaults to 7.5.
            save_path (str, optional): The path to save the generated images. Defaults to "output".

        """
        input_image_path = "input.png"
        init_image = self.get_image_from_url(image_url, input_image_path)
        init_images = [init_image for _ in range(batch_size)]
        for i in range(0, num_images, batch_size):
            with torch.xpu.amp.autocast(
                enabled=True if self.data_type != torch.float32 else False,
                dtype=self.data_type,
            ):
                if batch_size > 1:
                    inputs = self.get_inputs(batch_size=batch_size, prompt=prompt)
                    images = self.pipeline(
                        **inputs,
                        image=init_images,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    ).images
                else:
                    images = self.pipeline(
                        prompt=prompt,
                        image=init_images,
                        strength=strength,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                    ).images

                for j in range(len(images)):
                    output_image_path = os.path.join(
                        save_path,
                        f"{'_'.join(prompt.split()[:3])}_{i+j}.png",
                    )
                    images[j].save(output_image_path)


if __name__ == "__main__":
    output_dir = "output"
    num_images = 0
    model_ids = [
        "runwayml/stable-diffusion-v1-5",
        "stabilityai/stable-diffusion-2-1",
    ]
    print("Available models are:")
    for i, model_id in enumerate(model_ids):
        print(f"{i + 1}. {model_id}")
    try:
        selected_model_index = (
            int(
                input(
                    "Select a model by entering its number (or press enter to use the default model): "
                )
            )
            - 1
        )
    except ValueError:
        selected_model_index = 0
    model_id = (
        model_ids[selected_model_index]
        if 0 <= selected_model_index < len(model_ids)
        else model_ids[0]
    )
    prompt = input("Please enter your prompt: ")
    try:
        num_images = int(input("How many images have to be generated: (default 2) "))
    except Exception:
        num_images = 2
    if num_images <= 1:
        num_images = 2
    image_url = input("Please enter an image URL (or press Enter to use the default): ")
    if not image_url:
        print("The input is not a valid URL. Using the default URL instead.")
        image_url = "https://user-images.githubusercontent.com/786476/256401499-f010e3f8-6f8d-4e9f-9d1f-178d3571e7b9.png"
    elif not validators.url(image_url):
        print("The input is not a valid URL. Using the default URL instead.")
        image_url = "https://user-images.githubusercontent.com/786476/256401499-f010e3f8-6f8d-4e9f-9d1f-178d3571e7b9.png"
    model = Img2ImgModel(model_id, device="xpu")
    enhancements = [
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

    user_input = input(
        "Would you like to auto enhance the prompt futher? (yes/no): "
    ).lower()
    enhance = user_input in ["yes", "y", "1"]
    if enhance:
        prompt = prompt + " " + " ".join(random.sample(enhancements, 5))
        print(f"Using enhanced prompt: {prompt}")

    try:
        start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        model.generate_images(
            prompt=prompt,
            image_url=image_url,
            num_images=num_images,
        )
    except KeyboardInterrupt:
        print("\nUser interrupted image generation...")
    finally:
        print(
            f"Complete generating {num_images} image in {time.time() - start_time:.2f} seconds."
        )
