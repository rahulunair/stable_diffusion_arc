## Stable Diffusion inference on Intel Arc GPUs

[Now](https://blog.rahul.onl/posts/2022-08-12-arc-dgpu-linux.html) that we have our
Arc discrete GPU setup on Linux, let's try to run Stable Diffusion model using it.


## A quick recap / updated steps to set up Arc on Linux

Intel has now published [documentation](https://dgpu-docs.intel.com/installation-guides/ubuntu/ubuntu-jammy-arc.html) on how to set up Arc on Linux. 
I tried it today, it worked beautifully.

### Steps to configure Arc

- Install the 5.7 OEM kernel
- Install kernel mode drivers, gpu firmware
- Install usermod drivers for compute, 3d graphics and media
- Add user to `render` group
- Install oneAPI 2022.3 (latest as of this writeup)

## Stable Diffusion

Stable Diffusion is a fully open-source (thank you Stability.ai) deep learning text to image and image to image model. For more information on the model,
checkout the wikipedia [entry](https://en.wikipedia.org/wiki/Stable_Diffusion) for the same.

### PyTorch

To use PyTorch on Intel GPUs, we need to install, the Intel extensions for PyTorch or [ipex](https://github.com/intel/intel-extension-for-pytorch). Let's get the latest release
for [pyTorch](https://github.com/intel/intel-extension-for-pytorch/releases/download/v1.10.200%2Bgpu/torch-1.10.0a0+git3d5f2d4-cp39-cp39-linux_x86_64.whl) and [ipex](https://github.com/intel/intel-extension-for-pytorch/releases/download/v1.10.200%2Bgpu/intel_extension_for_pytorch-1.10.200+gpu-cp39-cp39-linux_x86_64.whl).

1. Create a conda environment with Python 3.9 and install both of the wheels.

```bash
~ → conda create -n ipex python=3.9 -y
```
```bash
~ → conda activate ipex
~ → pip install ~/Downloads/*.whl
```

Let's see how to run the model using PyTorch first,

2. Install diffusers library and dependencies


```bash
~ → pip install diffusers ftfy transformers Pillow
```

3. Run stable diffusion

We will use a model from 🤗 maintained by runwayml, `runwayml/stable-diffusion-v1-5`. To use the model, you will have to [generate](https://huggingface.co/docs/hub/security-tokens) a User access token for the 🤗 model hub.
Once generated we can easily download the model using diffusers API. Now that we have installed all the required packages and have the user token, lets try it out:


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

```python
In [8]: image = pipe(prompt).images[0]
   ...: 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:35<00:00,  1.43it/s]
In [9]: image = pipe(prompt).images[0]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 51/51 [00:09<00:00,  5.20it/s]
```

![](./images/sd_pyt_fp16.png)

As you can see the first time you run the model, it takes about 35 seconds, subsequent runs take about 10 seconds, you can expect this number to double when using fp32. 

### TensorFlow

Moving on to TensorFlow, we have this awesome repo from [divamgupta](https://github.com/divamgupta/stable-diffusion-tensorflow)

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

Executing this, we get the result:

```python
2022-11-06 23:00:51.948547: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
  0   1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:00<00:00,  1.21s/it]
2022-11-06 23:01:55.103111: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:114] Plugin optimizer for device_type XPU is enabled.
  0   1: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:29<00:00,  1.67it/s]
```

![](./images/sd_tf_fp32.png)

As you can see the first time you run the model, it takes about 60 seconds, subsequent runs take about 30 seconds. One thing to note here is that, for the TensorFlow version we used FP32 and not FP16 as in the case of pyTorch.
So a rough back of the envelope calculation suggests that the TF model is slightly faster than the pyTorch one.





