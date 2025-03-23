import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
import pickle
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import queue

# Import your SignRecognizer class
class SignRecognizer:
    def __init__(self, model_path=None, label_encoder_path=None, sequence_length=3, fps=15, 
                 confidence_threshold=30, detection_threshold=15):
        # MediaPipe setup
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Recognition parameters
        self.sequence_length = sequence_length
        self.fps = fps
        self.frames_per_sequence = sequence_length * fps
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold
        
        # Recognition state
        self.sequence_buffer = []
        self.current_sign = "No sign detected"
        self.confidence = 0.0
        self.no_detection_counter = 0
        self.is_recognizing = False
        
        # Model data
        self.model = None
        self.label_encoder = None
        
        # Load model and label encoder if paths provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        if label_encoder_path and os.path.exists(label_encoder_path):
            self.load_label_encoder(label_encoder_path)
            
    def load_model(self, model_path):
        """Load the trained model from file"""
        try:
            self.model = load_model(model_path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def load_label_encoder(self, label_encoder_path):
        """Load the label encoder from file"""
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading label encoder: {e}")
            return False
            
    def start_recognition(self):
        """Start the recognition process"""
        self.is_recognizing = True
        self.sequence_buffer.clear()
        self.no_detection_counter = 0
        self.current_sign = "No sign detected"
        self.confidence = 0.0
        
    def stop_recognition(self):
        """Stop the recognition process"""
        self.is_recognizing = False
        
    def extract_landmarks(self, results):
        """Extract hand and face landmarks from MediaPipe results"""
        landmarks = []
        # Extract left hand landmarks
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
            
        # Extract right hand landmarks
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)
            
        # Extract specific face landmarks
        face_indices = [0, 4, 8, 12, 14, 17, 21, 33, 37, 40, 43, 46, 49, 55, 69, 105, 127, 132, 148, 152]
        if results.face_landmarks:
            for idx in face_indices:
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * (len(face_indices) * 3))
            
        return landmarks
        
    def process_frame(self, frame):
        """Process a video frame for sign recognition"""
        if not self.is_recognizing or self.model is None or self.label_encoder is None:
            return frame, None
            
        # Prepare the frame for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(rgb_frame)
        
        # Create a copy for drawing
        display_frame = frame.copy()
        
        # Draw left hand landmarks with additional mesh lines
        if results.left_hand_landmarks:
            # Draw the standard connections
            self.mp_drawing.draw_landmarks(
                display_frame,
                results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            # Draw additional connections to create a more detailed mesh
            for i in range(5):  # For each finger
                base = i * 4 + 1  # Base index for each finger (excluding wrist)
                for j in range(base, base + 3):  # Connect adjacent points in the finger
                    if i < 4:  # Avoid connecting pinky to a non-existent next finger
                        cv2.line(
                            display_frame,
                            tuple(np.multiply(
                                [results.left_hand_landmarks.landmark[j].x, results.left_hand_landmarks.landmark[j].y],
                                [display_frame.shape[1], display_frame.shape[0]]
                            ).astype(int)),
                            tuple(np.multiply(
                                [results.left_hand_landmarks.landmark[j+4].x, results.left_hand_landmarks.landmark[j+4].y],
                                [display_frame.shape[1], display_frame.shape[0]]
                            ).astype(int)),
                            (0, 255, 0), 1
                        )
        
        # Draw right hand landmarks with additional mesh lines
        if results.right_hand_landmarks:
            # Draw the standard connections
            self.mp_drawing.draw_landmarks(
                display_frame,
                results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
            # Draw additional connections to create a more detailed mesh
            for i in range(5):  # For each finger
                base = i * 4 + 1  # Base index for each finger (excluding wrist)
                for j in range(base, base + 3):  # Connect adjacent points in the finger
                    if i < 4:  # Avoid connecting pinky to a non-existent next finger
                        cv2.line(
                            display_frame,
                            tuple(np.multiply(
                                [results.right_hand_landmarks.landmark[j].x, results.right_hand_landmarks.landmark[j].y],
                                [display_frame.shape[1], display_frame.shape[0]]
                            ).astype(int)),
                            tuple(np.multiply(
                                [results.right_hand_landmarks.landmark[j+4].x, results.right_hand_landmarks.landmark[j+4].y],
                                [display_frame.shape[1], display_frame.shape[0]]
                            ).astype(int)),
                            (0, 255, 0), 1
                        )
        
        # Draw face landmarks with tesselation style instead of contours
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(
                display_frame,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
        # Extract landmarks and check for valid hand data
        landmarks = self.extract_landmarks(results)
        has_landmarks = any(abs(x) > 0.001 for x in landmarks[:126])  # Check hand landmarks
        
        # Process for recognition
        recognition_result = None
        if has_landmarks:
            self.sequence_buffer.append(landmarks)
            self.no_detection_counter = 0
            
            # Maintain buffer size
            if len(self.sequence_buffer) > self.frames_per_sequence:
                self.sequence_buffer.pop(0)
                
            # Perform prediction if buffer is full
            if len(self.sequence_buffer) == self.frames_per_sequence:
                try:
                    X = np.array([self.sequence_buffer])
                    pred = self.model.predict(X, verbose=0)[0]
                    pred_idx = np.argmax(pred)
                    confidence = pred[pred_idx] * 100
                    debug_info = f"Buffer size: {len(self.sequence_buffer)}, Confidence: {confidence:.1f}%"
                    
                    # Return recognition result if confidence is high enough
                    if confidence > self.confidence_threshold:
                        sign = self.label_encoder.inverse_transform([pred_idx])[0]
                        self.current_sign = sign
                        self.confidence = confidence
                        recognition_result = {
                            'sign': sign, 
                            'confidence': confidence,
                            'debug': debug_info
                        }
                    else:
                        self.no_detection_counter += 1
                        recognition_result = {'debug': debug_info}
                except Exception as e:
                    print(f"Prediction error: {e}")
                    recognition_result = {'debug': f"Error: {str(e)}"}
        else:
            self.no_detection_counter += 1
            
        # Reset if no detection for too long
        if self.no_detection_counter > self.detection_threshold:
            self.current_sign = "No sign detected"
            self.confidence = 0
            self.sequence_buffer.clear()
            self.no_detection_counter = 0
            recognition_result = {'sign': "No sign detected", 'confidence': 0}
            
        return display_frame, recognition_result

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Initialize the recognizer
data_dir = "sign_language_data"
model_path = os.path.join(data_dir, "sign_model.h5")
label_encoder_path = os.path.join(data_dir, "label_encoder.pkl")
recognizer = SignRecognizer(model_path, label_encoder_path)

# Start recognition
recognizer.start_recognition()

@app.route('/api/recognize', methods=['POST'])
def recognize_sign():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Decode base64 image
        image_data = request.json['image'].split(',')[1] if ',' in request.json['image'] else request.json['image']
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Flip frame for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame for recognition
        display_frame, result = recognizer.process_frame(frame)
        
        # Encode the processed frame to return to client
        _, buffer = cv2.imencode('.jpg', display_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        response = {
            'processed_image': f'data:image/jpeg;base64,{processed_image}'
        }
        
        # Add recognition results if available
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

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)