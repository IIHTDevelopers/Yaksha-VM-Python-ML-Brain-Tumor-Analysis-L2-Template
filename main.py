import numpy as np
from tensorflow.keras.models import load_model
from dataloader import load_data, split_data, data_summary, get_image_count
from model_trainer import create_cnn_model, train_model, evaluate_model
import os

def get_image_count(data_path):
    """Counts the total number of images in the dataset."""
    print("get_image_count() called")
    return None

def preprocess_image(image_path, img_size=(64, 64)):
    """Preprocesses a single image for prediction."""
    print("preprocess_image() called")
    return None

def predict_test_folder(model_path, test_folder):
    """Predicts if images in the test_images folder have a tumor or not."""
    print("predict_test_folder() called")
    pass

if __name__ == "__main__":
    # Define paths
    DATA_PATH = "brain_tumor_dataset"
    MODEL_PATH = "brain_tumor_cnn_model.h5"
    TEST_FOLDER = "test_images"

    # Step 1: Load and preprocess the data
    print("Loading data...")
    X, y = load_data(DATA_PATH)

    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    data_summary(X_train, X_test, y_train, y_test)

    # Step 3: Create and train model
    print("Creating model...")
    model = create_cnn_model()
    print("Training model...")
    train_model(model, X_train, y_train)

    # Step 4: Save model
    print(f"Model saved to {MODEL_PATH}")

    # Step 5: Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Step 6: Count total images
    get_image_count(DATA_PATH)

    # Step 7: Predict images in test_images folder
    print("\nChecking test_images folder images...")
    predict_test_folder(MODEL_PATH, TEST_FOLDER)
