# Stable Diffusion Image Generation

This repository contains a script for generating images using the Stable Diffusion method. You can customize the model, the base image, the prompt, and the number of images to generate.

## Prerequisites

- Python 3.10
- PyTorch
- Transformers
- Diffusers
- Validators
- Intel Extension for PyTorch

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourgithubusername/yourrepositoryname.git
    ```
2. Change your directory to the cloned repository:
    ```bash
    cd yourrepositoryname
    ```
3. Install the necessary packages:
    ```bash
    pip install torch==1.10.1 transformers==5.2.2 diffusers==0.0.1 validators==0.18.2 intel-extension-for-pytorch==1.10.0
    ```

## Usage

To generate images using the Stable Diffusion method, run the `image_to_image.py` script:

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
