import os
import tensorflow as tf
from PIL import Image


def clean_corrupted_images(dataset_dirs):
    """Scans dataset directories and removes truncated or unreadable image files."""
    removed_count = 0
    for directory in dataset_dirs:
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}, skipping...")
            continue

        print(f"Scanning {directory} for corrupted files...")
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.load()
                        img.convert('RGB')
                except Exception:
                    print(f"Removing corrupted file: {file_path}")
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")
    print(f"Clean complete! Removed {removed_count} corrupted file(s).\n")


def load_datasets(train_dir, val_dir, test_dir, img_size=(224, 224), batch_size=32):
    """Loads train, validation, and test datasets with proper shuffle settings."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=False
    )
    return train_ds, val_ds, test_ds