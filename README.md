# Chest-CT-Scan-Image-Classification-using-Transfer-Learning-ResNet50-
Using ResNet50 transfer learning, this project classifies Chest CT-Scan images into four categories: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal. The model is trained and fine-tuned on a public Kaggle dataset, enabling accurate lung cancer detection with minimal data and computational resources.
This project classifies Chest CT-Scan images into four categories —
Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, and Normal — using transfer learning with a pretrained ResNet50 deep CNN model.

The model learns to detect lung cancer types based on features extracted from CT images. It was implemented in TensorFlow/Keras and trained in Google Colab using GPU acceleration.

📘 Dataset
Dataset: Chest CT-Scan Images Dataset – Kaggle (https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)

Contains train, test, and validation folders.

Each folder includes four sub-folders:

adenocarcinoma/

large cell carcinoma/

squamous cell carcinoma/

normal/

Images are CT scans of lungs showing different cancer types and healthy tissue.

⚙️ Model Description
Base Model: ResNet50 pretrained on ImageNet
Framework: TensorFlow / Keras
Input Shape: (224, 224, 3)
Trainable Layers: Last 30–60 layers unfrozen for fine-tuning
Optimizer: Adam (learning_rate = 1e-4 → 1e-5 after fine-tuning)
Loss: Categorical Cross-Entropy
Metrics: Accuracy, Precision, Recall, F1-Score
Epochs: 20–30 (with early stopping)
Batch Size: 32

Architecture Summary:


ResNet50 (frozen base)
 → GlobalAveragePooling2D
 → Dense(512, ReLU)
 → Dropout(0.4)
 → Dense(128, ReLU)
 → Dropout(0.3)
 → Dense(4, Softmax)
🧠 Training Pipeline
Data Loading & Augmentation using ImageDataGenerator
(rotation, zoom, brightness, horizontal flip, shift).

Transfer Learning: freeze base ResNet50 layers and train new classifier head.

Fine-Tuning: unfreeze top ResNet layers for medical-specific adaptation.

Evaluation: tested on separate validation and test splits.

Inference UI: single-image upload in Colab predicts the CT-Scan class.
