## Stable Diffusion inference on Intel Arc and Data Center GPUs 

<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/93824391-eaf9-4e5d-a3de-ea6c813cf255" width="400">|<img src="https://github.com/rahulunair/stable_diffusion_arc/assets/786476/f010e3f8-6f8d-4e9f-9d1f-178d3571e7b9" width="400">|


**Update**: If you want to benchmark stable diffusion on your Intel dGPUs and CPUs, checkout my other [repo](https://github.com/rahulunair/stable_diffusion_xpu) .

[Now](https://blog.rahul.onl/posts/2022-08-12-arc-dgpu-linux.html) that we have our Arc discrete GPU setup on Linux, let's try to run Stable Diffusion model using it. The gpu we are using is an Arc A770 16 GB card.


## A quick recap / updated steps to set up Arc (Intel dGPUs) on Linux

Please follow the [documentation](https://dgpu-docs.intel.com/driver/installation.html) on how to set up Intel dGPUs on Linux. After setting up your environment you can verify if everything works by running the python xpu_test.py script from the [xpu_verify](htttps://github.com/rahulunair/xpu_verify) repository.


## Stable Diffusion

Stable Diffusion is a fully open-source (thank you Stability.ai) deep learning text to image and image to image model. For more information on the model,
checkout the wikipedia [entry](https://en.wikipedia.org/wiki/Stable_Diffusion) for the same.

### PyTorch

To use PyTorch on Intel GPUs (xpus), we need to install, the Intel extensions for PyTorch or [ipex](https://github.com/intel/intel-extension-for-pytorch).

1. Create a conda environment with Python 3.9 and install both of the wheels.

```bash
~ → conda create -n ipex python=3.9 -y
```
```bash
~ → conda activate ipex
~ → python -m pip install torch==1.13.0a0+git6c9b55e intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
```

2. Install diffusers library and dependencies


```bash
~ → pip install invisible_watermark transformers accelerate safetensors
~ →  pip install diffusers --upgrade

```

3. Run stable diffusion

It is as simple as:

```bash
python sd_pytorch.py
```

### Supported Models:

```bash
1. CompVis/stable-diffusion-v1-4
2. stabilityai/stable-diffusion-2-1
3. stabilityai/stable-diffusion-xl-base-1.0
```

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
