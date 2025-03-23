import os
import base64
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

# Import your MongoDB helpers
from mongo_db import create_user, authenticate_user

# Import SignRecognizer from recog.py
from recog import SignRecognizer

app = Flask(__name__)
CORS(app)

# -------------------------
# Initialize SignRecognizer
# -------------------------
data_dir = "sign_language_data"
model_path = os.path.join(data_dir, "sign_model.h5")
label_encoder_path = os.path.join(data_dir, "label_encoder.pkl")

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

recognizer = SignRecognizer(model_path, label_encoder_path)
recognizer.start_recognition()

# -------------------------
# Sign recognition endpoints
# -------------------------
@app.route('/api/recognize', methods=['POST'])
def recognize_sign():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Decode base64 image
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Flip the frame for natural interaction
        frame = cv2.flip(frame, 1)

        # Process the frame using your recognizer
        display_frame, result = recognizer.process_frame(frame)

        # Encode the processed frame to send back to the client
        _, buffer = cv2.imencode('.jpg', display_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')

        response = {
            'processed_image': f'data:image/jpeg;base64,{processed_image}'
        }
        if result:
            if 'sign' in result:
                response['sign'] = result['sign']
                response['confidence'] = float(result['confidence'])
            if 'debug' in result:
                response['debug'] = result['debug']
        else:
            response['sign'] = "No sign detected"
            response['confidence'] = 0.0

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': recognizer.model is not None})

# -------------------------
# Authentication endpoints
# -------------------------
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400

    if create_user(username, email, password):
        return jsonify({"message": "User registered successfully"}), 201
    else:
        return jsonify({"error": "Username already exists"}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    is_valid = authenticate_user(username, password)

    if is_valid:
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

# -------------------------
# Run the app
# -------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
