import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Configuration Parameters
# -------------------------------
SEQUENCE_LENGTH = 16       # Number of frames to extract
IMG_SIZE = (224, 224)      # Image dimensions (must match training)
VIDEO_PATH = r"ProcessedDataset\NoAccident\negative_samples_1112.mp4"  # Replace with your video file path

# -------------------------------
# Helper Function: Extract Frames
# -------------------------------
def extract_frames(video_path, num_frames=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """
    Extracts 'num_frames' evenly spaced frames from the video at 'video_path'.
    Each frame is resized to 'img_size', converted to RGB, and normalized.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine indices for evenly spaced frames
    if total_frames < num_frames:
        indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frame_id = 0
    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in indices:
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        frame_id += 1
    cap.release()
    
    # Ensure we have exactly num_frames frames
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    frames = np.array(frames, dtype="float32") / 255.0
    return frames

# -------------------------------
# Load the Pre-Trained Model
# -------------------------------
# Update the model path if needed.
model = load_model("best_accident_detection_model.h5")

# -------------------------------
# Prepare Video Input for Prediction
# -------------------------------
frames = extract_frames(VIDEO_PATH)
# The model expects a batch dimension: (batch_size, SEQUENCE_LENGTH, height, width, channels)
video_input = np.expand_dims(frames, axis=0)

# -------------------------------
# Make a Prediction
# -------------------------------
prediction = model.predict(video_input)
print("Prediction (probability):", prediction[0][0])

# Use a threshold of 0.5 for binary classification
if prediction[0][0] > 0.5:
    print("Accident detected.")
else:
    print("No accident detected.")
