## Stable Diffusion inference on Intel Arc and Data Center GPUs

<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/36e194b1-8a17-47c7-89c3-995d989fae3f" text="A high-resolution, brightly colored, happy character in the style of Studio Jhibli art 4k gothic punchy dark" width="400">|<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/54621a2d-6306-4c30-b375-c8771de66ce4" width="400">
<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/93824391-eaf9-4e5d-a3de-ea6c813cf255" width="400">|<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/f010e3f8-6f8d-4e9f-9d1f-178d3571e7b9" width="400">
<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/b0a4a7a9-85c9-4480-98ae-12b06bdf37d5" text="A high-resolution, brightly colored, happy character from Studio Jibali  dreaming painting exquisite 4k detailed" width="400">|<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/99e2e01f-1d45-4924-8900-c14435bed385" width="400" text="A high resolution cartoon character with super powers night exquisite electric historical art">


The default implementation uses `PyTorch`, i have provided a TensorFlow version as well in the directoy `tensorflow`.

**Update**: If you want to benchmark stable diffusion on your Intel dGPUs and CPUs, checkout my other [repo](https://github.com/rahulunair/stable_diffusion_xpu) .

[Now](https://blog.rahul.onl/posts/2022-08-12-arc-dgpu-linux.html) that we have our Arc discrete GPU setup on Linux, let's try to run Stable Diffusion model using it. The gpu we are using is an Arc A770 16 GB card.


## A quick recap / updated steps to set up Arc (Intel dGPUs) on Linux

Please follow the [documentation](https://dgpu-docs.intel.com/driver/installation.html) on how to set up Intel dGPUs on Linux. After setting up your environment you can verify if everything works by running the python xpu_test.py script from the [xpu_verify](htttps://github.com/rahulunair/xpu_verify) repository.


## Stable Diffusion

There are two versions a **Text to image** version and an **image to image** version, `cd` to corresponding directories to get the code for each. For eg:

### Text-to-Image Model Using Stable Diffusion

Stable Diffusion is a fully open-source (thank you Stability.ai) deep learning text to image and image to image model. For more information on the model,
checkout the wikipedia [entry](https://en.wikipedia.org/wiki/Stable_Diffusion) for the same.

### PyTorch

To use PyTorch on Intel GPUs (xpus), we need to install, the Intel extensions for PyTorch or [ipex](https://github.com/intel/intel-extension-for-pytorch).
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

   Option 1:
   ```bash
   conda env create -f environment.yml
   ```
   Option 2:
    ```bash
    conda create -n diffusion_env python=3.10 -y
    conda activate diffusion_env
    pip install decorator
    # need the oneapi basekit >= 2023.2.0name: diffusion_env2
    pip install torch==2.0.1a0 torchvision==0.15.2a0 -f https://developer.intel.com/ipex-whl-stable-xpu
    pip install intel_extension_for_pytorch==2.0.110+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
    # if using oneapi baekit <= 2023.1.0
    # pip install  torch==1.13.0a0+git6c9b55e intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
    pip install acclerate transformers diffusers validators
    ```

5. Run stable diffusion

#### Monolithic version

The monolithic implementation can be found in the `sd_text_to_img.py` file. This version contains all necessary components, including model definition, loading, optimization, and inference code, in a single script. 
You can run the script from the command line as follows:

```bash
python sd_text_to_img.py
```

#### Client-server version

The client-server implementation splits the functionality into two separate scripts: sd_server.py for the server and sd_client.py for the client.

We have two additional requirements for this version:

```bash
pip install flask requests
```

To use this version, first start the server by running:

```bash
python sd_server.py
```
Then, in a separate terminal window, run the client script:

```bash
python sd_client.py
```

#### UI version

Run the server first:

```bash
python server.py
```

Then run gradio based ui web app:

```bash
python sd_ui.py
```

You should now be able to interact with a screen like this:

![image](https://github.com/rahulunair/stable_diffusion_arc/assets/786476/341f636d-04de-434f-be28-e5dc84e0dde9)


All versions offer the same functionality from a user's perspective. The monolithic version may be simpler to set up and run, as it doesn't require running two separate scripts. However, the client-server version could offer better performance for large-scale tasks, as it allows the server to handle multiple requests simultaneously.

### Supported Models:

```bash
1. CompVis/stable-diffusion-v1-4
2. stabilityai/stable-diffusion-2-1
3. stabilityai/stable-diffusion-xl-base-1.0
```

This repository contains two versions of a text-to-image model using Stable Diffusion: a monolithic implementation and a client-server implementation. Both versions offer a similar user interface, allowing you to choose the version that best suits your needs. 

### Gist on how to run on intel xpus 

### PyTorch

For the optimized version use `sd_pytorch.py`, but just to get an over all idea of what we are doing, here is the pytorch and tensorflow version:

```python
import intel_extension_for_pytorch
import torch
from diffusers import StableDiffusionPipeline

model_id="runwayml/stable-diffusion-v1-5"
prompt = "vivid red hot air ballons over paris in the evening"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,  # this can be torch.float32 as well
    revision="fp16",
    use_auth_token="<the token you generated>")
pipe = pipe.to("xpu")
image = pipe(prompt).images[0]
image.save(f"{prompt[:5]}.png")
```

Executing this, we get the result:

```bash
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 12.70it/s]
```


As you can see the first time you run the model, it takes about 35 seconds, subsequent runs take about 10 seconds, you can expect this number to double when using fp32. 

#### TensorFlow

Moving on to TensorFlow, we have this awesome repo from [divamgupta](https://github.com/divamgupta/stable-diffusion-tensorflow):

1. Create a conda environment with Python 3.9 and install tensorflow and intel_extension_for_tensorflow wheels.

```bash
~ → conda create -n itex python=3.9 -y
```

```bash
~ → conda activate itex
~ → pip install tensorflow==2.10.0
~ → pip install --upgrade intel-extension-for-tensorflow[gpu]
```

Let's see how to run the model using PyTorch first,

2. Install stable_diffusion_tensorflow package and dependencies


```bash
~ → pip install git+https://github.com/divamgupta/stable-diffusion-tensorflow ftfy pillow tqdm regex tensorflow-addons
```

3. Run stable diffusion

Running the TensorFlow model is straightforward as there are no user tokens or anything like that required.

```python
import intel_extension_for_tensorflow
import tensorflow
from stable_diffusion_tf.stable_diffusion import StableDiffusion
from PIL import Image

prompt = "vivid red hot air ballons over paris in the evening"
generator = StableDiffusion(
    img_height=512,
    img_width=512,
    jit_compile=False,
)

img = generator.generate(
    prompt,
    num_steps=50,
    unconditional_guidance_scale=7.5,
    temperature=1,
    batch_size=1,
)
Image.fromarray(img[0]).save("sd_tf_fp32.png")
```
