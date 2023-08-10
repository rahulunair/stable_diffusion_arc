from flask import Flask, request, jsonify
from sd_text_to_img import Text2ImgModel 

app = Flask(__name__)

@app.route('/generate_images', methods=['POST'])
def generate_images():
    data = request.json
    model_id = data.get('model_id')
    prompt = data.get('prompt')
    num_images = data.get('num_images', 1)
    model = Text2ImgModel(model_id, device="xpu")
    try:
        images = model.generate_images(
            prompt,
            num_images=num_images,
            save_path="./output",
        )
        image_paths = [img.filename for img in images]
        return jsonify({"status": "success", "image_paths": image_paths})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
