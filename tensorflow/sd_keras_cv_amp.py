# coding: utf-8
import intel_extension_for_tensorflow as itex
import keras_cv
import tensorflow as tf
from tensorflow import keras

auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()
auto_mixed_precision_options.data_type = itex.FLOAT16
graph_options = itex.GraphOptions(
    auto_mixed_precision_options=auto_mixed_precision_options
)
graph_options.auto_mixed_precision = itex.ON
config = itex.ConfigProto(graph_options=graph_options)
itex.set_backend("gpu", config)

model = keras_cv.models.StableDiffusion(img_width=512, img_height=512)

for _ in range(2):
    images = model.text_to_image(
        "photograph of an astronaut riding a horse", batch_size=1
    )

