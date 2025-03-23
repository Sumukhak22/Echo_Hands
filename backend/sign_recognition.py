import os
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox, ttk
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import time
from PIL import Image, ImageTk
from collections import deque
import threading
import queue
import pickle
import json
import datetime

# Global variables (previously instance variables)
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

sequence_length = 3
fps = 15
frames_per_sequence = sequence_length * fps
signs_data = {}
current_sequence = []
current_frames = []
model = None
label_encoder = None
sequence_buffer = []
current_sign = "No sign detected"
confidence = 0.0
no_detection_counter = 0
detection_threshold = 15
confidence_threshold = 30

root = None
video_label = None
is_capturing = False
is_recognizing = False
mode = "idle"
countdown_active = False
countdown_value = 0

data_dir = "sign_language_data"
model_path = os.path.join(data_dir, "sign_model.h5")
data_path = os.path.join(data_dir, "sign_data.pkl")
label_encoder_path = os.path.join(data_dir, "label_encoder.pkl")

frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue()

cap = None
is_running = False
video_thread = None

# UI-related global variables
sign_name_var = None
samples_var = None
capture_btn = None
train_btn = None
save_btn = None
load_btn = None
progress_frame = None
status_var = None
progress_var = None
progress_bar = None
countdown_var = None
countdown_label = None
recognize_btn = None
detected_sign_var = None
confidence_var = None
debug_var = None
signs_listbox = None
delete_sign_btn = None

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

def setup_ui():
    """Create the user interface"""
    global root, video_label, sign_name_var, samples_var, capture_btn, train_btn, save_btn, load_btn, \
           progress_frame, status_var, progress_var, progress_bar, countdown_var, countdown_label, \
           recognize_btn, detected_sign_var, confidence_var, debug_var, signs_listbox, delete_sign_btn, \
           cap, is_running, video_thread

    root = tk.Tk()
    root.title("Sign Language Detection")
    root.geometry("1024x768")

    # Main frames
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

    # Video display
    video_frame = ttk.LabelFrame(left_frame, text="Camera Feed")
    video_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    video_label = ttk.Label(video_frame)
    video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Training section
    training_frame = ttk.LabelFrame(right_frame, text="Training")
    training_frame.pack(fill=tk.X, pady=5)

    ttk.Label(training_frame, text="Sign Name:").pack(anchor=tk.W, padx=5, pady=5)
    sign_name_var = tk.StringVar()
    ttk.Entry(training_frame, textvariable=sign_name_var).pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(training_frame, text="Samples to Collect:").pack(anchor=tk.W, padx=5, pady=5)
    samples_var = tk.IntVar(value=5)
    ttk.Entry(training_frame, textvariable=samples_var).pack(fill=tk.X, padx=5, pady=5)

    btn_frame = ttk.Frame(training_frame)
    btn_frame.pack(fill=tk.X, padx=5, pady=5)

    capture_btn = ttk.Button(btn_frame, text="Capture Sign", command=start_capture)
    capture_btn.pack(side=tk.LEFT, padx=5)

    train_btn = ttk.Button(btn_frame, text="Train Model", command=train_model, state=tk.DISABLED)
    train_btn.pack(side=tk.LEFT, padx=5)

    # Save/Load data buttons
    save_load_frame = ttk.Frame(training_frame)
    save_load_frame.pack(fill=tk.X, padx=5, pady=5)

    save_btn = ttk.Button(save_load_frame, text="Save Data", command=save_data)
    save_btn.pack(side=tk.LEFT, padx=5)

    load_btn = ttk.Button(save_load_frame, text="Load Data", command=load_data)
    load_btn.pack(side=tk.LEFT, padx=5)

    # Training progress
    progress_frame = ttk.LabelFrame(right_frame, text="Progress")
    progress_frame.pack(fill=tk.X, pady=5)

    status_var = tk.StringVar(value="Ready")
    ttk.Label(progress_frame, textvariable=status_var).pack(anchor=tk.W, padx=5, pady=5)

    progress_var = tk.DoubleVar(value=0)
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var, maximum=100)
    progress_bar.pack(fill=tk.X, padx=5, pady=5)

    countdown_var = tk.StringVar(value="")
    countdown_label = ttk.Label(progress_frame, textvariable=countdown_var, font=("Arial", 16, "bold"))
    countdown_label.pack(padx=5, pady=5)

    # Recognition section
    recognition_frame = ttk.LabelFrame(right_frame, text="Recognition")
    recognition_frame.pack(fill=tk.X, pady=5)

    recognize_btn = ttk.Button(recognition_frame, text="Start Recognition", command=toggle_recognition, state=tk.DISABLED)
    recognize_btn.pack(fill=tk.X, padx=5, pady=5)

    ttk.Label(recognition_frame, text="Detected Sign:").pack(anchor=tk.W, padx=5, pady=5)
    detected_sign_var = tk.StringVar(value="No sign detected")
    ttk.Label(recognition_frame, textvariable=detected_sign_var, font=("Arial", 16, "bold")).pack(padx=5, pady=5)

    confidence_var = tk.DoubleVar(value=0)
    ttk.Progressbar(recognition_frame, variable=confidence_var, maximum=100).pack(fill=tk.X, padx=5, pady=5)

    debug_var = tk.StringVar(value="")
    ttk.Label(recognition_frame, textvariable=debug_var, font=("Arial", 8)).pack(padx=5, pady=5)

    # Collected signs list
    signs_list_frame = ttk.LabelFrame(right_frame, text="Collected Signs")
    signs_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

    signs_listbox = tk.Listbox(signs_list_frame)
    signs_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    listbox_btn_frame = ttk.Frame(signs_list_frame)
    listbox_btn_frame.pack(fill=tk.X, padx=5, pady=5)

    delete_sign_btn = ttk.Button(listbox_btn_frame, text="Delete Sign", command=delete_selected_sign)
    delete_sign_btn.pack(side=tk.LEFT, padx=5)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        root.destroy()
        return

    # Try to load existing data and model
    try:
        load_data()
    except Exception as e:
        print(f"No existing data found or error loading data: {e}")

    # Start video thread
    is_running = True
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()

    # Start UI update
    update_ui()

    root.protocol("WM_DELETE_WINDOW", on_close)

def process_video():
    """Process video frames in a separate thread"""
    global is_running, cap, frame_queue, result_queue, mode, is_capturing, countdown_active, \
           current_sequence, current_frames, frames_per_sequence, progress_var, sequence_buffer, \
           no_detection_counter, model, label_encoder

    prev_frame_time = 0
    while is_running:
        curr_time = time.time()
        elapsed = curr_time - prev_frame_time
        if elapsed < 1.0 / fps:
            time.sleep(max(0, 1.0 / fps - elapsed))
        prev_frame_time = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)
        display_frame = frame.copy()

        # DRAW HAND LANDMARKS
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # DRAW FACE MESH (FULL TESSELLATION)
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Use face_mesh for iris landmarks
            mp_drawing.draw_landmarks(
                display_frame,
                results.face_landmarks,
                mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

        # DRAW POSE LANDMARKS (FULL BODY)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                display_frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style(),
                mp_drawing_styles.get_default_pose_connections_style()
            )

        if countdown_active:
            cv2.putText(
                display_frame,
                f"{countdown_value}",
                (display_frame.shape[1] // 2 - 50, display_frame.shape[0] // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8
            )

        if mode == "training" and is_capturing and not countdown_active:
            landmarks = _extract_landmarks(results)
            current_sequence.append(landmarks)
            current_frames.append(frame.copy())
            sequence_progress = min(100, (len(current_sequence) / frames_per_sequence) * 100)
            progress_var.set(sequence_progress)
            progress_text = f"Recording: {len(current_sequence)}/{frames_per_sequence} frames"
            cv2.putText(display_frame, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if len(current_sequence) >= frames_per_sequence:
                is_capturing = False
                _save_captured_sequence()

        elif mode == "recognizing" and is_recognizing:
            landmarks = _extract_landmarks(results)
            has_landmarks = any(abs(x) > 0.001 for x in landmarks[:126])
            if has_landmarks:
                sequence_buffer.append(landmarks)
                no_detection_counter = 0
                if len(sequence_buffer) > frames_per_sequence:
                    sequence_buffer.pop(0)
                if len(sequence_buffer) == frames_per_sequence and model is not None:
                    try:
                        X = np.array([sequence_buffer])
                        pred = model.predict(X, verbose=0)[0]
                        pred_idx = np.argmax(pred)
                        confidence = pred[pred_idx] * 100
                        debug_info = f"Buffer size: {len(sequence_buffer)}, Confidence: {confidence:.1f}%"
                        result_queue.put({'debug': debug_info})
                        if confidence > confidence_threshold:
                            sign = label_encoder.inverse_transform([pred_idx])[0]
                            result_queue.put({'sign': sign, 'confidence': confidence})
                        else:
                            no_detection_counter += 1
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        result_queue.put({'debug': f"Error: {str(e)}"})
            else:
                no_detection_counter += 1
            if no_detection_counter > detection_threshold:
                result_queue.put({'sign': "No sign detected", 'confidence': 0})
                sequence_buffer.clear()
                no_detection_counter = 0

        if not frame_queue.full():
            frame_queue.put(display_frame)

def start_countdown(callback):
    """Start a countdown before capturing a sequence"""
    global countdown_active, countdown_value, countdown_var
    countdown_active = True
    countdown_value = 3

    def update_countdown():
        global countdown_active, countdown_value
        if countdown_value > 0:
            countdown_var.set(f"Get ready! {countdown_value}")
            countdown_value -= 1
            root.after(1000, update_countdown)
        else:
            countdown_var.set("GO!")
            countdown_active = False
            callback()
            root.after(1000, lambda: countdown_var.set(""))

    update_countdown()

def update_ui():
    """Update UI elements with latest data"""
    global video_label, frame_queue, result_queue, detected_sign_var, confidence_var, debug_var
    try:
        if not frame_queue.empty():
            frame = frame_queue.get_nowait()
            _update_video_display(frame)
        while not result_queue.empty():
            result = result_queue.get_nowait()
            if 'sign' in result:
                detected_sign_var.set(result['sign'])
                confidence_var.set(result['confidence'])
            if 'debug' in result:
                debug_var.set(result['debug'])
    except queue.Empty:
        pass
    except Exception as e:
        print(f"UI update error: {e}")
    root.after(30, update_ui)

def _update_video_display(frame):
    """Update the video display with the current frame"""
    global video_label
    h, w = frame.shape[:2]
    max_h, max_w = 480, 640
    if h > max_h or w > max_w:
        scale = min(max_h / h, max_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

def _extract_landmarks(results):
    """Extract hand and face landmarks from MediaPipe results"""
    landmarks = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    face_indices = [0, 4, 8, 12, 14, 17, 21, 33, 37, 40, 43, 46, 49, 55, 69, 105, 127, 132, 148, 152]
    if results.face_landmarks:
        for idx in face_indices:
            lm = results.face_landmarks.landmark[idx]
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * (len(face_indices) * 3))
    return landmarks

def start_capture():
    """Start capturing frames for a sign sequence"""
    global is_capturing, mode, current_sequence, current_frames, capture_btn, signs_data, signs_listbox, status_var
    sign_name = sign_name_var.get().strip().upper()
    if not sign_name:
        messagebox.showerror("Error", "Please enter a sign name")
        return
    capture_btn.config(state=tk.DISABLED)
    current_sequence = []
    current_frames = []
    if sign_name not in signs_data:
        signs_data[sign_name] = []
        signs_listbox.insert(tk.END, sign_name)
    mode = "training"
    status_var.set(f"Preparing to capture sign: {sign_name}...")
    start_countdown(_begin_capture)

def _begin_capture():
    """Begin capturing sequence after countdown"""
    global is_capturing, status_var
    is_capturing = True
    sign_name = sign_name_var.get().strip().upper()
    status_var.set(f"Capturing sign: {sign_name}...")

def _save_captured_sequence():
    """Save the captured sequence to the signs data"""
    global is_capturing, current_sequence, signs_data, status_var, progress_var, capture_btn, train_btn
    sign_name = sign_name_var.get().strip().upper()
    if len(current_sequence) > frames_per_sequence:
        current_sequence = current_sequence[:frames_per_sequence]
    signs_data[sign_name].append(current_sequence)
    samples_count = len(signs_data[sign_name])
    total_samples = samples_var.get()
    status_var.set(f"Saved sequence {samples_count}/{total_samples} for {sign_name}")
    progress_var.set((samples_count / total_samples) * 100)
    capture_btn.config(state=tk.NORMAL)
    if samples_count < total_samples:
        messagebox.showinfo("Sequence Captured",
                            f"Successfully captured sequence {samples_count}/{total_samples} for {sign_name}.\n"
                            f"Click 'Capture Sign' to record the next sequence.")
    else:
        status_var.set(f"Completed capturing {sign_name}")
        progress_var.set(0)
        messagebox.showinfo("Sign Completed",
                            f"Successfully captured all {total_samples} sequences for {sign_name}.\n"
                            f"You can now capture another sign or train the model.")
        save_data()
    if len(signs_data) >= 2:
        train_btn.config(state=tk.NORMAL)

def delete_selected_sign():
    """Delete the selected sign from the dataset"""
    global signs_data, signs_listbox, train_btn
    selection = signs_listbox.curselection()
    if not selection:
        messagebox.showinfo("Information", "Please select a sign to delete")
        return
    sign_name = signs_listbox.get(selection[0])
    if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete all data for '{sign_name}'?"):
        if sign_name in signs_data:
            del signs_data[sign_name]
        signs_listbox.delete(selection[0])
        if len(signs_data) < 2:
            train_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Success", f"Sign '{sign_name}' has been deleted")
        save_data()

def save_data():
    """Save the collected sign data to a file"""
    global signs_data, model, label_encoder, status_var
    if not signs_data:
        messagebox.showinfo("Information", "No data to save")
        return
    try:
        with open(data_path, 'wb') as f:
            pickle.dump(signs_data, f)
        metadata = {
            'sign_names': list(signs_data.keys()),
            'frames_per_sequence': frames_per_sequence,
            'date_saved': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_samples': sum(len(samples) for samples in signs_data.values())
        }
        with open(os.path.join(data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=4)
        if model is not None:
            model.save(model_path)
        if label_encoder is not None:
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(label_encoder, f)
        status_var.set(f"Saved data: {metadata['total_samples']} samples across {len(signs_data)} signs")
        messagebox.showinfo("Success", f"Data saved successfully to {data_dir}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save data: {str(e)}")

def load_data():
    """Load sign data and model from files"""
    global signs_data, model, label_encoder, signs_listbox, train_btn, recognize_btn, status_var
    try:
        if not os.path.exists(data_path):
            messagebox.showinfo("Information", "No saved data found")
            return
        with open(data_path, 'rb') as f:
            signs_data = pickle.load(f)
        signs_listbox.delete(0, tk.END)
        for sign_name in signs_data:
            signs_listbox.insert(tk.END, sign_name)
        if os.path.exists(model_path):
            model = load_model(model_path)
            if os.path.exists(label_encoder_path):
                with open(label_encoder_path, 'rb') as f:
                    label_encoder = pickle.load(f)
            recognize_btn.config(state=tk.NORMAL)
        if len(signs_data) >= 2:
            train_btn.config(state=tk.NORMAL)
        if os.path.exists(os.path.join(data_dir, 'metadata.json')):
            with open(os.path.join(data_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            status_var.set(f"Loaded data: {metadata.get('total_samples', 0)} samples across {len(signs_data)} signs")
        else:
            total_samples = sum(len(samples) for samples in signs_data.values())
            status_var.set(f"Loaded data: {total_samples} samples across {len(signs_data)} signs")
        messagebox.showinfo("Success", f"Data loaded successfully from {data_dir}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {str(e)}")

def _build_model(input_shape, num_classes):
    """Build the LSTM model for sign recognition"""
    model = Sequential([
        LSTM(64, return_sequences=True, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape),
        Dropout(0.3),
        LSTM(128, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'),
        Dropout(0.3),
        LSTM(64, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def train_model():
    """Train the model on collected sign sequences"""
    global signs_data, model, label_encoder, progress_var, status_var, recognize_btn, capture_btn, train_btn
    if len(signs_data) < 2:
        messagebox.showerror("Error", "Please collect at least 2 different signs")
        return
    capture_btn.config(state=tk.DISABLED)
    train_btn.config(state=tk.DISABLED)
    X, y = [], []
    status_var.set("Preparing training data...")
    progress_var.set(0)
    for i, (sign, sequences) in enumerate(signs_data.items()):
        for sequence in sequences:
            if len(sequence) == frames_per_sequence:
                X.append(sequence)
                y.append(sign)
        progress = ((i + 1) / len(signs_data)) * 40
        progress_var.set(progress)
        root.update()
    if not X:
        messagebox.showerror("Error", "No valid sequences found for training")
        capture_btn.config(state=tk.NORMAL)
        train_btn.config(state=tk.NORMAL)
        return
    X = np.array(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    status_var.set("Building and training model...")
    progress_var.set(40)
    root.update()
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    input_shape = (X.shape[1], X.shape[2])
    num_classes = len(label_encoder.classes_)
    if os.path.exists(model_path) and os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, 'rb') as f:
                saved_label_encoder = pickle.load(f)
            if np.array_equal(saved_label_encoder.classes_, label_encoder.classes_):
                model = load_model(model_path)
                status_var.set("Loaded existing model for further training")
            else:
                status_var.set("Sign classes have changed. Creating new model.")
                model = _build_model(input_shape, num_classes)
        except Exception as e:
            status_var.set(f"Error loading existing data: {e}. Creating new model.")
            model = _build_model(input_shape, num_classes)
    else:
        model = _build_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    progress_var.set(50)
    root.update()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return float(lr)
        return float(lr * tf.math.exp(-0.1))
    lr_callback = LearningRateScheduler(lr_scheduler)
    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    try:
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            callbacks=[early_stopping, lr_callback],
            verbose=1
        )
        progress_var.set(100)
        status_var.set("Model training completed!")
        save_data()
        recognize_btn.config(state=tk.NORMAL)
        val_accuracy = history.history['val_categorical_accuracy'][-1] * 100
        messagebox.showinfo("Training Complete", f"Model trained successfully with {val_accuracy:.2f}% validation accuracy.")
    except Exception as e:
        messagebox.showerror("Training Error", f"Error during model training: {str(e)}")
    finally:
        capture_btn.config(state=tk.NORMAL)
        train_btn.config(state=tk.NORMAL)

def toggle_recognition():
    """Toggle between recognition mode on/off"""
    global is_recognizing, mode, recognize_btn, status_var, capture_btn, detected_sign_var, confidence_var, sequence_buffer
    if is_recognizing:
        is_recognizing = False
        mode = "idle"
        recognize_btn.config(text="Start Recognition")
        status_var.set("Recognition stopped")
        capture_btn.config(state=tk.NORMAL)
    else:
        if model is None:
            messagebox.showerror("Error", "No trained model available. Please train a model first.")
            return
        is_recognizing = True
        mode = "recognizing"
        recognize_btn.config(text="Stop Recognition")
        status_var.set("Recognition active...")
        capture_btn.config(state=tk.DISABLED)
        detected_sign_var.set("No sign detected")
        confidence_var.set(0)
        sequence_buffer.clear()

def on_close():
    """Cleanup when closing the application"""
    global is_running, cap, signs_data
    is_running = False
    if cap is not None:
        cap.release()
    try:
        if signs_data:
            save_data()
    except Exception as e:
        print(f"Error saving data on exit: {e}")
    root.destroy()

def main():
    """Main entry point"""
    setup_ui()
    root.mainloop()

if __name__ == "__main__":
    main()