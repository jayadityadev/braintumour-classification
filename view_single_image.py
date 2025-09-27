# view_single_image.py

import matplotlib.pyplot as plt
from scripts.data_loader import load_image_data

# --- Configuration ---
# You can change the number to view any image from your dataset.
IMAGE_TO_VIEW = 'data/15.mat' 

# --- Load the Data ---
print(f"Loading data from {IMAGE_TO_VIEW}...")
data = load_image_data(IMAGE_TO_VIEW)

# --- Visualize the Image and Mask ---
if data:
    image = data['image']
    tumor_mask = data['tumorMask'] # 

    # Create a plot with two panels, side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the original MRI image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original MRI Image')
    axes[0].axis('off')
    
    # Display the ground truth tumor mask
    axes[1].imshow(tumor_mask, cmap='gray')
    axes[1].set_title('Ground Truth Tumor Mask')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
else:
    print("Could not load or display the image.")