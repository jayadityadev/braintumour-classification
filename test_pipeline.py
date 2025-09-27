# test_pipeline.py

import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

# Import functions from your scripts
from scripts.data_loader import load_image_data
from scripts.enhancement_pipeline import apply_wiener_filter, apply_ica_for_contrast

# --- 1. Configuration ---
IMAGE_PATH = 'data/15.mat' # Use a sample image from your data folder
MODEL_PATH = 'models/denoising_autoencoder.keras'

# --- 2. Load Tools ---
print("Loading the trained autoencoder model...")
autoencoder = load_model(MODEL_PATH)
print("Loading a sample image...")
original_image = load_image_data(IMAGE_PATH)['image']

# In test_pipeline.py

# --- 3. Run the Full Pipeline ---
print("Running the full enhancement pipeline...")

# --- NEW NORMALIZATION STEP ---
# Normalize the original image to the 0.0 to 1.0 range
original_normalized = original_image.astype(np.float32) / original_image.max()

# Step 1: Wiener Filter (on normalized image)
wiener_filtered = apply_wiener_filter(original_normalized)

# Step 2: Denoising Autoencoder
# Model expects a 4D tensor: (batch_size, height, width, channels)
model_input = np.expand_dims(wiener_filtered, axis=0)
model_input = np.expand_dims(model_input, axis=-1)
autoencoder_output = autoencoder.predict(model_input)
autoencoder_denoised = autoencoder_output.squeeze()

# Step 3: ICA for Contrast
ica_enhanced = apply_ica_for_contrast(autoencoder_denoised)

# --- 4. Visualize and Confirm ---
print("Displaying results...")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('1. Original Image')
axes[0].axis('off')

axes[1].imshow(wiener_filtered, cmap='gray')
axes[1].set_title('2. After Wiener Filter')
axes[1].axis('off')

axes[2].imshow(autoencoder_denoised, cmap='gray')
axes[2].set_title('3. After Autoencoder')
axes[2].axis('off')

axes[3].imshow(ica_enhanced, cmap='gray')
axes[3].set_title('4. Final ICA Enhanced')
axes[3].axis('off')

plt.tight_layout()
plt.show()