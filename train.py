import os
from src.utils import clean_corrupted_images, load_datasets
from src.model import build_cnn_model

def main():
    print("=== Bone Fracture Classification Pipeline ===")

    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    test_dir = 'dataset/test'

    clean_corrupted_images([train_dir, val_dir, test_dir])

    print("Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(train_dir, val_dir, test_dir)

    print("Building CNN model architecture...")
    model = build_cnn_model()
    model.summary()

    print("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    print("Evaluating model on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")

    os.makedirs('saved_models', exist_ok=True)
    model_path = 'saved_models/bone_fracture_cnn_model.keras'
    model.save(model_path)
    print(f"Model saved successfully to {model_path}!")

if __name__ == '__main__':
    main()