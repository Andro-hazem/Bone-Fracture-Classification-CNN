import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_datasets


def main():
    print("=== Bone Fracture Model Evaluation ===")

    # 1. Define paths
    test_dir = 'dataset/test'
    model_path = 'saved_models/bone_fracture_cnn_model.keras'

    # 2. Check if the saved model exists
    if not os.path.exists(model_path):
        print(f"Error: No saved model found at {model_path}.")
        print("Please run train.py first or make sure your .keras file is in place.")
        return

    # 3. Load the saved model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

    # 4. Load test dataset
    print("Loading test dataset...")
    # Note: We only need the test dataset here, but load_datasets returns all three
    _, _, test_ds = load_datasets(
        train_dir='dataset/train',
        val_dir='dataset/val',
        test_dir=test_dir
    )

    # Extract class names from the directory structure
    class_names = test_ds.class_names
    print(f"Classes detected: {class_names}")

    # 5. Evaluate overall loss and accuracy
    print("\nRunning evaluation on test set...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # 6. Generate Classification Report & Confusion Matrix
    print("\nGenerating classification report...")
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


if __name__ == '__main__':
    main()