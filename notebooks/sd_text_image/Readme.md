## How to Run the Jupyter Notebooks

### Version 1

- [Monolithic Code](./stable_diffusion_text_to_image.ipynb): This notebook contains all the necessary components, including model definition, loading, optimization, and inference code. You can execute each cell one by one to generate images. If you encounter issues, such as the notebook kernel being killed, please refer to the 'Client-Server Code' section below and use the `sd_server.py` and `sd_client.py` code.

- [Client-Server Code](./sd_client_server.ipynb): To run the client-server code, first open a Jupyter notebook. Then, from the notebook interface, open a terminal window and switch to the `diffusion_xpu` conda environment. Run the server by entering the following command:

```bash
# set up conda env and install required pkgs.

#  conda create -n diffusion_xpu python=3.10 -y
# conda activate diffusion_xpu

# python -m pip install torch==1.13.0a0+git6c9b55e intel_extension_for_pytorch==1.13.120+xpu -f https://developer.intel.com/ipex-whl-stable-xpu
# pip install flask
#  pip install invisible_watermark transformers accelerate safetensors
# pip install diffusers --upgrade

python server.py
```

Once the server is up and running successfully, you can proceed with the cells in the notebook. You will be able to choose the model, enter a prompt, specify the number of images, and view the generated images.
