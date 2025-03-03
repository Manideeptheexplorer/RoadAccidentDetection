import os
import random
import math
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2

# ===================================
# Hyperparameters and Configurations
# ===================================
SEQUENCE_LENGTH = 16       # Number of frames per video clip
IMG_SIZE = (224, 224)      # Image dimensions for each frame (compatible with MobileNetV2)
BATCH_SIZE = 8             # Batch size for training
EPOCHS = 30                # Maximum number of training epochs
LEARNING_RATE = 1e-4       # Learning rate for optimizer
L2_REG = 1e-3              # L2 regularization factor
AUGMENT = True             # Enable data augmentation in the training generator

# ===================================
# 1. Load File Paths from Folders
# ===================================
# Directories for Accident and NoAccident videos (already divided)
accident_dir = "ProcessedDataset/Accident"      # Accident videos folder
noaccident_dir = "ProcessedDataset/NoAccident"    # NoAccident videos folder

# Get list of video files (adjust file extensions as needed)
accident_files = [os.path.join(accident_dir, f) for f in os.listdir(accident_dir)
                  if f.lower().endswith(('.mp4', '.avi', '.mov'))]
noaccident_files = [os.path.join(noaccident_dir, f) for f in os.listdir(noaccident_dir)
                    if f.lower().endswith(('.mp4', '.avi', '.mov'))]

print("Total Accident videos:", len(accident_files))
print("Total NoAccident videos:", len(noaccident_files))

# Create combined lists with labels (Accident = 1, NoAccident = 0)
all_files = accident_files + noaccident_files
all_labels = [1] * len(accident_files) + [0] * len(noaccident_files)

# ===================================
# 2. Split Data into Train+Validation and Test Sets
# ===================================
# First, split the dataset into train_val (80%) and test (20%) sets.
train_val_files, test_files, train_val_labels, test_labels = train_test_split(
    all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)

print("\nTotal for training+validation:", len(train_val_files))
print("Total for testing (unseen):", len(test_files))

# Further split train_val into training (80% of train_val) and validation (20% of train_val)
train_files, val_files, train_labels, val_labels = train_test_split(
    train_val_files, train_val_labels, test_size=0.2, random_state=42, stratify=train_val_labels)

print("\nBefore oversampling (Training set):")
print("  Train Accident videos:", sum(1 for l in train_labels if l == 1))
print("  Train NoAccident videos:", sum(1 for l in train_labels if l == 0))
print("Validation set:")
print("  Validation Accident videos:", sum(1 for l in val_labels if l == 1))
print("  Validation NoAccident videos:", sum(1 for l in val_labels if l == 0))
print("Test set:")
print("  Test Accident videos:", sum(1 for l in test_labels if l == 1))
print("  Test NoAccident videos:", sum(1 for l in test_labels if l == 0))

# ===================================
# 3. Oversample the Minority Class in the Training Set
# ===================================
# Separate training file paths by label
train_accident_files = [f for f, l in zip(train_files, train_labels) if l == 1]
train_noaccident_files = [f for f, l in zip(train_files, train_labels) if l == 0]

num_accident = len(train_accident_files)
num_noaccident = len(train_noaccident_files)

# Oversample the minority (NoAccident) if needed
if num_noaccident < num_accident:
    extra_needed = num_accident - num_noaccident
    extra_noaccident_files = np.random.choice(train_noaccident_files, size=extra_needed, replace=True).tolist()
    train_files += extra_noaccident_files
    train_labels += [0] * extra_needed

# Shuffle the oversampled training set
combined_train = list(zip(train_files, train_labels))
random.shuffle(combined_train)
train_files, train_labels = zip(*combined_train)

print("\nAfter oversampling (Training set):")
print("  Train Accident videos:", sum(1 for l in train_labels if l == 1))
print("  Train NoAccident videos:", sum(1 for l in train_labels if l == 0))

# ===================================
# 4. Define Data Augmentation & Data Generator
# ===================================
def augment_frames(frames):
    """
    Apply random augmentations to a sequence of frames.
    Example augmentations: random horizontal flip and brightness adjustment.
    """
    # Random horizontal flip with probability 0.5
    if random.random() < 0.5:
        frames = np.flip(frames, axis=2)  # Flip along the width axis
    # Random brightness adjustment
    brightness_factor = random.uniform(0.8, 1.2)
    frames = np.clip(frames * brightness_factor, 0, 1)
    return frames

def extract_frames(video_path, num_frames=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """
    Extracts 'num_frames' evenly spaced frames from the video.
    Resizes each frame to 'img_size', converts to RGB, and normalizes pixel values.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    
    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])
    
    frames = np.array(frames, dtype="float32") / 255.0
    return frames

class VideoDataGenerator(Sequence):
    """
    Keras Sequence generator that yields batches of video clips and labels.
    For training, data augmentation can be applied.
    """
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, 
                 augment=False, shuffle=True):
        self.file_paths = list(file_paths)
        self.labels = list(labels)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.augment = augment
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
            if self.augment:
                frames = augment_frames(frames)
            batch_X.append(frames)
        batch_X = np.array(batch_X)  # Shape: (batch_size, sequence_length, height, width, channels)
        batch_y = np.array(batch_labels)
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            combined = list(zip(self.file_paths, self.labels))
            random.shuffle(combined)
            self.file_paths, self.labels = zip(*combined)

# Create generators: apply augmentation for training only
train_gen = VideoDataGenerator(train_files, train_labels, batch_size=BATCH_SIZE, 
                               sequence_length=SEQUENCE_LENGTH, augment=AUGMENT, shuffle=True)
val_gen = VideoDataGenerator(val_files, val_labels, batch_size=BATCH_SIZE, 
                             sequence_length=SEQUENCE_LENGTH, augment=False, shuffle=False)
test_gen = VideoDataGenerator(test_files, test_labels, batch_size=BATCH_SIZE, 
                              sequence_length=SEQUENCE_LENGTH, augment=False, shuffle=False)

# ===================================
# 5. Build the CNN+LSTM Model with Regularization
# ===================================
def build_model(sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE):
    """
    Builds a CNN+LSTM model.
    MobileNetV2 is used as the feature extractor (TimeDistributed),
    followed by an LSTM layer and a Dense output layer.
    L2 regularization is applied to reduce overfitting.
    """
    input_shape = (sequence_length, img_size[0], img_size[1], 3)
    video_input = Input(shape=input_shape)
    
    # Base model: MobileNetV2 (pre-trained on ImageNet)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False  # Freeze base model
    
    x = TimeDistributed(base_model)(video_input)
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    x = Dropout(0.5)(x)
    
    # LSTM layer with L2 regularization
    x = LSTM(64, kernel_regularizer=l2(L2_REG), return_sequences=False)(x)
    x = Dropout(0.5)(x)
    
    # Dense output layer with L2 regularization
    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(L2_REG))(x)
    
    model = Model(inputs=video_input, outputs=output)
    return model

# Build and compile the model
model = build_model(sequence_length=SEQUENCE_LENGTH, img_size=IMG_SIZE)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
model.summary()

# ===================================
# 6. Train the Model with Early Stopping and Model Checkpoint
# ===================================
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("best_accident_detection_model.h5", monitor='val_loss', save_best_only=True)

model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, 
          callbacks=[early_stopping, model_checkpoint])

# ===================================
# 7. Evaluate on Unseen Test Data
# ===================================
loss, accuracy = model.evaluate(test_gen)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Save final model if needed
model.save("final_accident_detection_model.h5")
print("Model saved as 'final_accident_detection_model.h5'")
