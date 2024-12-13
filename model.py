from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model():
    """
    Builds a Convolutional Neural Network (CNN) model.

    Instructions:
    1. Add convolutional layers using Conv2D with specified filters, kernel size, and activation.
    2. Include pooling layers (e.g., MaxPooling2D) to reduce spatial dimensions.
    3. Use Flatten to convert the 2D feature maps to a 1D vector.
    4. Add dense (fully connected) layers for classification.
    5. Use Dropout layers to reduce overfitting (optional).
    6. Compile the model with an optimizer (e.g., 'adam'), loss function (e.g., 'binary_crossentropy'), and metrics (e.g., 'accuracy').

    Returns:
        Sequential: A compiled CNN model.
    """
    pass  # Placeholder for building the CNN model
