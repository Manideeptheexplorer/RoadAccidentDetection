import os
import math
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# --------------------------------
# Configuration Parameters
# --------------------------------
SEQUENCE_LENGTH = 16       # Number of frames per video clip
IMG_SIZE = (224, 224)      # Frame dimensions (compatible with MobileNetV2)
BATCH_SIZE = 8             # Batch size for testing

# --------------------------------
# 1. Load File Paths from Folders
# --------------------------------
# Directories for Accident and NoAccident videos
accident_dir = "ProcessedDataset/Accident"
noaccident_dir = "ProcessedDataset/NoAccident"

# Get list of video files (adjust file extensions as needed)
accident_files = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir)
                  if f.lower().endswith(('.mp4', '.avi', '.mov'))]
noaccident_files = [os.path.join(noaccident_dir, f) for f in os.listdir(noaccident_dir)
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))]

# Combine file paths and labels (Accident = 1, NoAccident = 0)
all_files = accident_files + noaccident_files
all_labels = [1] * len(accident_files) + [0] * len(noaccident_files)

print("Total Accident videos:", len(accident_files))
print("Total NoAccident videos:", len(noaccident_files))
print("Total videos:", len(all_files))

# --------------------------------
# 2. Split into Train+Validation and Test Sets
# --------------------------------
# Since you don't have a separate test folder, we use a 20% test split.
_, test_files, _, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

print("\nTotal Test Videos:", len(test_files))
print("Test Accident videos:", sum(1 for l in test_labels if l == 1))
print("Test NoAccident videos:", sum(1 for l in test_labels if l == 0))

# --------------------------------
# 3. Define Helper Function to Extract Frames
# --------------------------------
def extract_frames(video_path, num_frames=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """
    Extracts 'num_frames' evenly spaced frames from the video at 'video_path'.
    Resizes frames to 'img_size', converts to RGB, and normalizes pixel values.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate indices: if video has fewer frames, duplicate the last frame as needed
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
    
    # Ensure we have the required number of frames
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    frames = np.array(frames, dtype="float32") / 255.0
    return frames

# --------------------------------
# 4. Define Test Data Generator
# --------------------------------
class VideoDataGenerator(Sequence):
    """
    Keras Sequence generator that yields batches of video clips and labels.
    No augmentation is applied in the test generator.
    """
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, shuffle=False):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return math.ceil(len(self.file_paths) / self.batch_size)
    
    def __getitem__(self, idx):
        batch_paths = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = []
        for file_path in batch_paths:
            frames = extract_frames(file_path, num_frames=self.sequence_length, img_size=IMG_SIZE)
            batch_X.append(frames)
        batch_X = np.array(batch_X)  # Shape: (batch_size, sequence_length, height, width, channels)
        batch_y = np.array(batch_labels)
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.file_paths, self.labels))
            np.random.shuffle(combined)
            self.file_paths, self.labels = zip(*combined)

# Create the test data generator
test_gen = VideoDataGenerator(test_files, test_labels, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, shuffle=False)
print("Number of test batches:", len(test_gen))

# --------------------------------
# 5. Load the Trained Model and Compute Metrics
# --------------------------------
# Load your saved model (update the filename if necessary)
model = load_model("final_accident_detection_model.h5")

# Collect true labels and predicted labels from the test generator
y_true = []
y_pred = []

for i in range(len(test_gen)):
    batch_X, batch_y = test_gen[i]
    preds = model.predict(batch_X)
    # Convert predicted probabilities to binary predictions using threshold 0.5
    preds_binary = (preds > 0.5).astype(int).flatten()
    y_true.extend(batch_y)
    y_pred.extend(preds_binary)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --------------------------------
# 6. Log Additional Metrics
# --------------------------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["NoAccident", "Accident"]))

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1-Score:", f1_score(y_true, y_pred))
