# scripts/enhancement_pipeline.py

import numpy as np
from scipy.signal import wiener
from sklearn.decomposition import FastICA
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

# ----------------------------------------------------------------------------
# ## FUNCTION 1: WIENER FILTER FOR NOISE REDUCTION
# ----------------------------------------------------------------------------

# In scripts/enhancement_pipeline.py

# In scripts/enhancement_pipeline.py

def apply_wiener_filter(image):
    """
    Applies a Wiener filter. Assumes input is a normalized float image.
    """
    # This function now expects a float image (e.g., values from 0.0 to 1.0)
    filtered_image = wiener(image, mysize=5)
    
    # Clean up any potential NaN values from flat areas
    np.nan_to_num(filtered_image, copy=False, nan=0.0)
    
    return filtered_image

# ----------------------------------------------------------------------------
# ## FUNCTION 2: AUTOENCODER MODEL FOR ADVANCED DENOISING
# ----------------------------------------------------------------------------

def create_denoising_autoencoder(input_shape=(512, 512, 1)):
    """
    Defines the architecture for a denoising convolutional autoencoder.
    This function BUILDS the model but does not TRAIN it. Training will be
    done in a separate script.
    
    Args:
        input_shape (tuple): The shape of the input images.
        
    Returns:
        Model: A compiled TensorFlow/Keras model.
    """
    # Encoder compresses the image into a smaller summary.
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder reconstructs the clean image from the summary.
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create and compile the model.
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder

# ----------------------------------------------------------------------------
# ## FUNCTION 3: ICA FOR CONTRAST ENHANCEMENT
# ----------------------------------------------------------------------------

def apply_ica_for_contrast(image, n_components=3):
    """
    Applies Independent Component Analysis (ICA) and returns the second 
    component, which enhances tumor contrast as per the base paper.
    
    Args:
        image (np.array): The input image from the previous step.
        n_components (int): The number of components for ICA to find.
        
    Returns:
        np.array: The contrast-enhanced image.
    """
    # Reshape image from 2D to 1D for ICA processing.
    pixel_list = image.reshape(-1, 1)
    
    # Create slightly varied features for ICA to work effectively.
    features = np.c_[pixel_list, 0.95 * pixel_list, 0.90 * pixel_list]

    # Initialize and run FastICA.
    ica = FastICA(n_components=n_components, random_state=0, whiten='unit-variance')
    independent_components = ica.fit_transform(features)
    
    # As per the paper, the second component (index 1) provides the best contrast.
    main_component = independent_components[:, 1]
    
    # Reshape the component back to the original image shape.
    ica_image = main_component.reshape(image.shape)
    
    # Normalize the pixel values to the standard 0-255 range.
    ica_image = 255 * (ica_image - ica_image.min()) / (ica_image.max() - ica_image.min())
    
    return ica_image.astype(image.dtype)