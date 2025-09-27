# scripts/data_loader.py

import h5py
import numpy as np
import os

def load_image_data(file_path):
    """
    Loads a v7.3 .mat file and extracts the image, label, and mask using h5py.
    
    Args:
        file_path (str): The full path to the .mat file.
        
    Returns:
        dict: A dictionary containing the 'image', 'label', and 'tumorMask',
              or None if an error occurs.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Data from MATLAB v7.3 files is often transposed. 
            # We use .T to correct the orientation.
            image = np.array(f.get('cjdata/image')).T
            label = int(f.get('cjdata/label')[0][0])
            tumor_mask = np.array(f.get('cjdata/tumorMask')).T
        
        return {
            'image': image,
            'label': label,
            'tumorMask': tumor_mask
        }

    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading {os.path.basename(file_path)}: {e}")
        return None