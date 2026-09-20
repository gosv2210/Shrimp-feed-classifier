# Shrimp Feed Classification using ResNet50

A deep learning-based image classification system for identifying shrimp feed levels from images using a transfer learning approach with **ResNet50**.

## Overview

This project uses an ImageNet-pretrained **ResNet50** convolutional neural network to classify shrimp images into five feed-level categories:

* **0%** — No feed
* **25%** — 25% feed level
* **50%** — 50% feed level
* **75%** — 75% feed level
* **100%** — 100% feed level

The ResNet50 backbone is used as a frozen feature extractor, while a custom classification head is trained for the shrimp feed classification task.

## Model Architecture

The model is based on **ResNet50 pretrained on ImageNet** with the original classification head removed.

The custom classification head consists of:

```text
Input Image (224 × 224 × 3)
          │
          ▼
     ResNet50
   (ImageNet weights)
   Frozen backbone
          │
          ▼
Global Average Pooling
          │
          ▼
Dense Layer (256)
     ReLU activation
          │
          ▼
Dropout (0.5)
          │
          ▼
Dense Layer (5)
   Softmax activation
          │
          ▼
Predicted Feed Level
```

## Dataset

The dataset is organized into five class directories:

```text
augmented_ds/
├── 0/
├── 25/
├── 50/
├── 75/
└── 100/
```

Each directory contains images corresponding to its respective feed-level class.

The dataset is split into:

* **70% Training**
* **15% Validation**
* **15% Testing**

A fixed `random_state=42` is used for the train/validation/test split.

> The dataset itself is not included in this repository.

## Image Preprocessing

Images are processed before being passed to ResNet50:

* Resized to **224 × 224**
* Converted to 3-channel RGB images
* ResNet50-specific `preprocess_input` normalization is applied
* Batch size: **32**

TensorFlow `tf.data.Dataset` is used for efficient loading, batching, and prefetching.

## Training

The ResNet50 backbone is initialized with ImageNet pretrained weights and frozen during training.

Training configuration:

| Parameter          | Value                           |
| ------------------ | ------------------------------- |
| Backbone           | ResNet50                        |
| Pretrained weights | ImageNet                        |
| Input size         | 224 × 224 × 3                   |
| Number of classes  | 5                               |
| Dense layer        | 256 units                       |
| Dropout            | 0.5                             |
| Optimizer          | Adam                            |
| Learning rate      | 0.001                           |
| Loss               | Sparse Categorical Crossentropy |
| Batch size         | 32                              |
| Epochs             | 20                              |

The trained model is saved as:

```text
resnet50_shrimp_feed_model.h5
```

## Evaluation

The notebook evaluates the trained classifier on the held-out test set using several metrics.

### Classification Metrics

* Test Accuracy
* Precision
* Recall
* Macro F1 Score
* Per-class F1 Score
* Confusion Matrix
* Average Precision (AP) for each class
* Mean Average Precision (mAP)

Training and validation accuracy/loss are also plotted to analyze model convergence and potential overfitting.

## Test Image Prediction

The notebook includes an inference section that allows a user to provide an image path and obtain the predicted feed level.

The inference pipeline:

```text
Input Image
     │
     ▼
Resize to 224 × 224
     │
     ▼
ResNet50 Preprocessing
     │
     ▼
Trained ResNet50 Model
     │
     ▼
Predicted Class
     │
     ▼
0% / 25% / 50% / 75% / 100%
```

The prediction can also be visualized directly on the input image along with the original and predicted labels.

## Test Set Generation

The notebook also contains functionality to create a separate test-set directory while preserving the original class structure:

```text
new_test_split/
├── 0/
├── 25/
├── 50/
├── 75/
└── 100/
```

This makes it easier to inspect and reuse the held-out test images.

## Running the Notebook

The notebook was developed for **Google Colab** and uses Google Drive to access the dataset.

### 1. Open the notebook

Open:

```text
_resNet_Shrimp_classifier_new.ipynb
```

in Google Colab or Jupyter Notebook.

### 2. Mount Google Drive

The notebook mounts Google Drive using:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Prepare the dataset

Place the dataset in the expected directory:

```text
/content/drive/MyDrive/Projdsnew/augmented_ds/
```

with the five class folders:

```text
0/
25/
50/
75/
100/
```

### 4. Train the model

Run the training section of the notebook. The model will be trained for 20 epochs and saved as:

```text
resnet50_shrimp_feed_model.h5
```

### 5. Evaluate

Run the evaluation cells to generate:

* Test accuracy
* Classification report
* Confusion matrix
* Precision
* Recall
* F1 scores
* AP per class
* mAP

### 6. Run inference

Load the saved model and provide the path to an image when prompted:

```text
Enter image path:
```

The notebook will display the image with its original and predicted feed-level labels.

## Dependencies

The project uses:

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* scikit-learn

The notebook is designed to run directly in Google Colab, where these dependencies are generally available.

## Repository Structure

```text
Shrimp-feed-classifier/
│
├── _resNet_Shrimp_classifier_new.ipynb
└── README.md
```

## Future Improvements

Potential extensions to the current implementation include:

* Fine-tuning the deeper ResNet50 layers after initial training
* Expanding the dataset with additional shrimp feeding conditions
* Applying stronger data augmentation
* Comparing ResNet50 with other CNN architectures
* Deploying the trained model for real-time shrimp feed-level prediction
* Evaluating robustness on images captured under different environmental and lighting conditions

## License

This project is intended for academic and research purposes.
