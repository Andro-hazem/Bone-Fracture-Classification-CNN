# 🦴 Bone Fracture Classification using CNN

A deep learning pipeline built with TensorFlow/Keras for binary classification of medical X-ray images into **Fractured** and **Non-Fractured** categories. This repository features a modularized Python architecture, exploratory notebooks, dataset cleaning scripts, evaluation tools, and pre-trained model weights.

---

## 📌 Project Overview
* **Objective:** Automatically detect bone fractures from medical X-ray images to assist diagnostic workflows.
* **Architecture:** Custom Convolutional Neural Network (CNN) optimized with `GlobalAveragePooling2D` for high accuracy and a lightweight footprint.
* **Model Size:** ~5 MB (enabling efficient memory usage and direct version tracking).
* **Key Features:**
  * Automated pre-processing to detect and purge corrupted or truncated image files.
  * Modular package structure (`src/`) for data loading, preprocessing, and model creation.
  * Standalone training (`train.py`) and evaluation (`evaluate.py`) pipelines.
  * Interactive exploratory data analysis in Jupyter Notebooks.

---

## 📁 Repository Structure

```text
bone-fracture-classifier/
├── dataset/                    <-- Expected dataset location (ignored by Git)
├── notebooks/                  <-- Interactive explorations & visual reports
│   └── bone_fracture_classifier.ipynb
├── saved_models/               <-- Pre-trained model weights
│   └── bone_fracture_cnn_model.keras
├── src/                        <-- Core modular Python package
│   ├── __init__.py
│   ├── model.py                <-- CNN architecture definition
│   └── utils.py                <-- Data loading and cleaning helpers
├── .gitignore                  <-- Git exclusion rules
├── evaluate.py                 <-- Model evaluation script
├── requirements.txt            <-- Project dependencies
└── train.py                    <-- End-to-end training pipeline script
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Andro-hazem/Bone-Fracture-Classification-CNN
   cd bone-fracture-classifier
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   # On Windows:
   python -m venv .venv
   .venv\Scripts\activate

   # On macOS/Linux:
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Setup

Due to GitHub file size limits, the raw X-ray image dataset is not tracked directly in this repository.

1. Download your bone fracture X-ray dataset from https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data.
2. Extract the contents into the root `dataset/` folder so it matches this structure:
   ```text
   dataset/
   ├── train/
   ├── val/
   └── test/
   ```

---

## 🚀 How to Run

### 1. Evaluate Pre-trained Weights
To immediately test the pre-trained 5 MB model on your test dataset without retraining:
```bash
python evaluate.py
```

### 2. Train Model from Scratch
To run the full cleaning, loading, training, and saving pipeline:
```bash
python train.py
```

### 3. Interactive Notebook
To explore the data visually, check distributions, and review training plots, open the notebook:
```bash
jupyter notebook notebooks/bone_fracture_classifier.ipynb
```

---

## 📈 Model Architecture & Performance
* **Input Shape:** $224 \times 224 \times 3$
* **Pooling:** `GlobalAveragePooling2D`
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy