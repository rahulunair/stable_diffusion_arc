# Stable Diffusion Image to Image Generation

This repository contains a script for generating images using the Stable Diffusion method. You can customize the model, the base image, the prompt, and the number of images to generate.

## Prerequisites

- Python 3.10
- PyTorch
- Intel extension for PyTorch
- Transformers
- Accelerate
- Diffusers
- Validators
- Intel Extension for PyTorch

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/rahulunair/stable_diffusion_arc
    ```
2. Change your directory to the cloned repository:
   
    ```bash
    cd stable_diffusion_arc
    ```
4. Install the necessary packages:
   
    ```bash
    conda create -n diffusion_env python=3.10 -y
    conda activate diffusion_env
    pip install decorator
    # need the oneapi basekit >= 2023.2.0
    #pip install torch==2.0.1a0 torchvision==0.15.2a0 -f https://developer.intel.com/ipex-whl-stable-xpu
    #pip install intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
    # if using oneapi baekit <= 2023.1.0
    pip install  torch==1.13.0a0+git6c9b55e intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
    pip install acclerate transformers diffusers validators
    ```

## Usage

To generate images using the Stable Diffusion method, run the `image_to_image.py` script:

Set up oneAPI sycl and MKL runtime using:

```bash
source /opt/intel/oneapi/setvars.sh
```

Activate python environment you setup:

```bash
conda activate diffusion_env
```

```bash
python image_to_image.py
```

You will be prompted to:

 - Select a model from the list of available models
 - Provide a prompt for the image generation
 - Specify the number of images to generate
 - Provide a URL for the base image (or use the default image)
 - Decide whether or not to enhance the prompt

 ## Example

 ```bash
 python image_to_image.py 
```

## Output

```bash
Available models are:
1. runwayml/stable-diffusion-v1-5
2. stabilityai/stable-diffusion-2-1

Select a model by entering its number (or press enter to use the default model): 2

Please enter your prompt: a more beautiful version of this utra focus

How many images have to be generated: (default 2) 4

Please enter an image URL (or press Enter to use the default): https://user-images.githubusercontent.com/786476/257653599-54621a2d-6306-4c30-b375-c8771de66ce4.png
...
Complete generating 4 images in 9.72 seconds.
```
