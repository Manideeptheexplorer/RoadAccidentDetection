import os
import shutil

# Define the source directory containing the 12 folders
source_dir = "Video-Accident-Dataset"  # Change this path to your dataset folder

# Define the destination directory for reorganized data
destination_dir = "ProcessedDataset"
accident_dest = os.path.join(destination_dir, "Accident")
no_accident_dest = os.path.join(destination_dir, "NoAccident")

# Create destination folders if they don't exist
os.makedirs(accident_dest, exist_ok=True)
os.makedirs(no_accident_dest, exist_ok=True)

print("Reorganizing dataset into two folders: Accident and NoAccident...")

# Iterate over each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        # Check if this folder represents no accident videos
        if folder_name.lower() == "negative_samples":
            dest_folder = no_accident_dest
        else:
            dest_folder = accident_dest

        # Iterate over files in the current folder (adjust extensions as needed)
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.mp4', '.avi', '.mov')):
                src_file = os.path.join(folder_path, file_name)
                dest_file = os.path.join(dest_folder, file_name)
                shutil.copy(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

print("Dataset reorganization complete.")
print(f"Total Accident videos: {len(os.listdir(accident_dest))}")
print(f"Total NoAccident videos: {len(os.listdir(no_accident_dest))}")
