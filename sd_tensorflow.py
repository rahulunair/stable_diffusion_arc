import os

import intel_extension_for_tensorflow as itex
import tensorflow as tf
from PIL import Image
from stable_diffusion_tf.stable_diffusion import StableDiffusion


def set_backend(backend="GPU", mixed=False):
    """auto mixed precision or fp32 mode"""
    if mixed:
        auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
        auto_mixed_precision_options.data_type = itex.FLOAT16
        graph_options = itex.GraphOptions(
            auto_mixed_precision_options=auto_mixed_precision_options
        )
        graph_options.auto_mixed_precision = itex.ON
        config = itex.ConfigProto(graph_options=graph_options)
        itex.set_backend(backend, config)
    else:
        itex.set_backend(backend)


set_backend()
prompt = "Red air balloons in the blue sky evening golden rays from the sun paris"
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
)[0]
Image.fromarray(img).save("./sd_tf_fp32.png")
