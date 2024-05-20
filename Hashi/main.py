from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
import io, os
import model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_MLIR_OPTIMIZATIONS'] = '1'

@app.route('/predictLeaves', methods=['POST'])
def predictLeaves():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image"})
    
    image_data = image_file.read()
    
    try:
        result = model.process_image(image_data, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/predictTree', methods=['POST'])
def predictTree():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected image"})
    
    image_data = image_file.read()
    
    try:
        result = model.process_image(image_data, 2)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({"error": "working"})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
