# scripts/train_autoencoder.py

import os
import glob
import numpy as np
import sys
from data_loader import load_image_data
from enhancement_pipeline import create_denoising_autoencoder, apply_wiener_filter

# --- 1. Configuration (More Robust Pathing) ---
# Get the absolute path of the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the project's root directory (which is one level up from 'scripts')
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# Define the absolute paths for data and model saving
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODEL_SAVE_PATH = os.path.join(ROOT_DIR, 'models', 'denoising_autoencoder.keras')

TRAINING_IMAGE_COUNT = 500
EPOCHS = 10

# --- 2. Load and Prepare Data ---
print(f"Searching for .mat files in: {DATA_DIR}")
all_image_paths = glob.glob(os.path.join(DATA_DIR, '*.mat'))

# --- ADDED SANITY CHECK ---
if not all_image_paths:
    print("\n--- ERROR ---")
    print(f"No .mat files were found in the specified directory.")
    print("Please make sure your 3064 .mat files are located directly inside:")
    print(f"'{DATA_DIR}'")
    sys.exit() # Exit the script cleanly

# (This code replaces the section above)
print(f"Found {len(all_image_paths)} images. Using {TRAINING_IMAGE_COUNT} for training.")
training_paths = all_image_paths[:TRAINING_IMAGE_COUNT]

raw_images_data = [load_image_data(p) for p in training_paths]

# --- START DEBUGGING SECTION ---
print("\n--- Starting Image Processing in Debug Mode ---")
clean_images_list = []
for i, data in enumerate(raw_images_data):
    if data is None:
        print(f"!! SKIPPING: Image at path {training_paths[i]} failed to load.")
        continue

    img = data['image']
    print(f"Processing image {i+1}/{len(raw_images_data)} (Path: {os.path.basename(training_paths[i])})")
    
    # 1. Check the raw image
    if not isinstance(img, np.ndarray) or img.shape != (512, 512):
        print(f"  - ERROR: Raw image is invalid! Shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
        continue

    # 2. Apply the filter
    filtered_img = apply_wiener_filter(img)
    
    # 3. Check the filtered image for problems
    contains_nan = np.isnan(filtered_img).any()
    print(f"  - Filtered. Shape: {filtered_img.shape}, Dtype: {filtered_img.dtype}, Contains NaN: {contains_nan}")

    if contains_nan:
        print(f"  - WARNING: NaN values detected in this image AFTER filtering. This is a potential problem.")
        
    clean_images_list.append(filtered_img)

print("\n--- Processing complete. Attempting to create final NumPy array. ---")
try:
    clean_images = np.array(clean_images_list)
    print("Successfully created NumPy array for clean_images.")
except Exception as e:
    print("\n--- CRITICAL ERROR ---")
    print("Failed to create NumPy array from the list of processed images.")
    print(f"Error: {e}")
    sys.exit() # Stop the script
# --- END DEBUGGING SECTION ---

# The rest of the script continues from here...
noise_factor = 0.2
noisy_images = clean_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_images.shape)
# ... etc.
noisy_images = clean_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_images.shape)
clean_images = np.clip(clean_images, 0., 255.) / 255.
noisy_images = np.clip(noisy_images, 0., 255.) / 255.
clean_images = np.expand_dims(clean_images, axis=-1)
noisy_images = np.expand_dims(noisy_images, axis=-1)

# --- 3. Create and Train the Model ---
print("Creating and training the autoencoder model...")
autoencoder = create_denoising_autoencoder()
autoencoder.fit(noisy_images, clean_images,
                epochs=EPOCHS,
                batch_size=16,
                shuffle=True,
                validation_split=0.1)

# --- 4. Save the Trained Model ---
print(f"Training complete. Saving model to {MODEL_SAVE_PATH}...")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True) # Ensure 'models' directory exists
autoencoder.save(MODEL_SAVE_PATH)
print("Model saved successfully!")