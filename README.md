# 🎀 Breast Cancer Histopathology Image Classification using CNNs

This repository contains the coursework and code for a deep learning project focused on the automatic detection and classification of breast cancer using histopathology images.

## 🌟 Introduction

[cite_start]Breast cancer is one of the most frequent and severe malignancies affecting women worldwide and remains a leading cause of cancer-related mortality[cite: 6, 7]. [cite_start]Early detection is crucially important as it improves treatment efficacy, increases survival rates, and reduces patient difficulties[cite: 9].

[cite_start]This project addresses a **binary classification** challenge: training a computer model to distinguish visual characteristics between **benign** (not cancerous) and **malignant** (cancerous) breast tissue[cite: 15, 16, 17]. [cite_start]This automated "second opinion" can help reduce the workload and potential human error for pathologists [cite: 25, 26, 12][cite_start], and aid in closing the expertise gap in emerging nations[cite: 29].

### Why Deep Learning (CNNs)?
[cite_start]Deep learning, particularly Convolutional Neural Networks (CNNs), has altered the processing of medical pictures[cite: 32]. [cite_start]CNNs are extremely effective in detecting cancer because they can automatically identify patterns in photos without requiring human feature design[cite: 33, 34]. 

The CNN learns features critical for cancer detection, such as:
* [cite_start]The texture of cells[cite: 36].
* [cite_start]Changes in colour[cite: 37].
* [cite_start]Tissue structural alterations (e.g., asymmetrical forms or dense clusters of cells)[cite: 38].

---

## 📂 Dataset Overview

The model was trained and validated on a large, balanced dataset of breast tissue histopathology images.

* [cite_start]**Total Images:** 10,000 photos across the collection[cite: 49].
* [cite_start]**Image Dimensions:** All photos were downsized to $512 \times 512$ pixels, using three colour channels (RGB)[cite: 60].
* [cite_start]**Classification Task:** Binary (Benign / Malignant)[cite: 53].
* [cite_start]**Dataset Balance:** The dataset is properly balanced, with an equal number of photos in both categories[cite: 88, 89].

### Dataset Distribution

[cite_start]The dataset is cleanly split into training and testing folders, with labels automatically generated from the folder names[cite: 54].

| Folder Path | Class | Images in Training | Images in Testing | Total |
| :--- | :--- | :--- | :--- | :--- |
| `.../train/breast_benign` | Benign | [cite_start]4000 [cite: 48] | - | 4000 |
| `.../train/breast_malignant` | Malignant | [cite_start]4000 [cite: 48] | - | 4000 |
| `.../test/breast_benign` | Benign | - | [cite_start]1000 [cite: 48] | 1000 |
| `.../test/breast_malignant` | Malignant | - | [cite_start]1000 [cite: 48] | 1000 |

### Preprocessing and Augmentation

[cite_start]The `ImageDataGenerator` was configured to load images in batches and perform several transformations[cite: 81]. [cite_start]This process enhances the pictures and strengthens the model, reducing the likelihood of memorizing the training data[cite: 83, 84].

**Augmentation Parameters:**
* [cite_start]Rescale: `1./255` (Normalizing pixel values) [cite: 81]
* [cite_start]`rotation_range=20` [cite: 82]
* [cite_start]`shear_range=0.2` [cite: 82]
* [cite_start]`zoom_range=0.2` [cite: 82]
* [cite_start]`horizontal_flip=True` [cite: 82]
* [cite_start]`validation_split=0.2` (Sets aside 20% of training data for validation) [cite: 85]

The resulting data generators found:
* **Training Set Size:** 6400 images
* **Validation Set Size:** 1600 images

---

## 🧠 Model Architecture (Sequential CNN)

The model is a custom Sequential Convolutional Neural Network designed for this classification task.

| Layer (Type) | Output Shape | Parameters | Activation Function |
| :--- | :--- | :--- | :--- |
| `conv2d` (Conv2D) | (None, 510, 510, 32) | 896 | `relu` |
| `max_pooling2d` (MaxPooling2D) | (None, 255, 255, 32) | 0 | - |
| `conv2d_1` (Conv2D) | (None, 253, 253, 64) | 18,496 | `relu` |
| `max_pooling2d_1` (MaxPooling2D) | (None, 126, 126, 64) | 0 | - |
| `conv2d_2` (Conv2D) | (None, 124, 124, 128) | 73,856 | `relu` |
| `max_pooling2d_2` (MaxPooling2D) | (None, 62, 62, 128) | 0 | - |
| `flatten` (Flatten) | (None, 492032) | 0 | - |
| `dense` (Dense) | (None, 256) | 125,960,448 | `sigmoid` |
| `dense_1` (Dense) | (None, 1) | 257 | `sigmoid` |
| **Total Trainable Params** | | **126,053,953** | |

### Training Configuration
* **Loss Function:** `'binary_crossentropy'`
* **Optimizer:** `'adam'`
* **Metrics:** `'accuracy'`

---

## 🚀 Usage

### Prerequisites

To run the notebook, you will need the following Python libraries:
* `tensorflow` / `keras`
* `numpy`
* `matplotlib`
* `opencv` (`cv2`)

### Running the Code

1.  **Clone the repository:**
    ```bash
    git clone [Your Repository Link]
    ```
2.  **Organize the Dataset:**
    Ensure your dataset is compressed into a zip file named `Breast_Cancer.zip` and placed in an accessible location (e.g., in a Google Drive folder).
3.  **Execute the Notebook:**
    The entire project flow is contained within the `FinalProject (1).ipynb` notebook. Execute the cells sequentially, particularly those for mounting the drive and unzipping the dataset.

### Making Predictions

Once the model is trained, you can use the following steps to make a prediction on a single image:

1.  Load the image using OpenCV.
2.  Preprocess the image: convert BGR to RGB, resize to $(512, 512)$, normalize pixel values by dividing by 255.0.
3.  Add a batch dimension: `np.expand_dims(img_resized, axis=0)`.
4.  Use `model.predict(img_tensor)` to get the output probability.
5.  Classify the result: If the probability is greater than $0.5$, the prediction is **Malignant**; otherwise, it is **Benign**.

The trained model is intended to be saved as a Keras H5 file: `/content/drive/MyDrive/Breast_Cancer_Model.h5`.

---

## 📈 Results and Future Work

The model training was initiated for 2 epochs but did not complete the second epoch.

* **Epoch 1 Performance:**
    * Training Accuracy: $0.5011$
    * Validation Accuracy: $0.5000$
    * Training Loss: $1.8225$
    * Validation Loss: $0.6932$

The initial results show the model just starting the learning process.

### Recommended Next Steps

1.  **Complete Training:** The model needs to be run for a sufficient number of epochs (e.g., 10-20) to ensure full convergence and utilization of the training data.
2.  **Evaluation:** Perform a complete evaluation on the 2000-image test set to determine the final, unbiased performance metrics.
3.  **Architecture Improvement:** Experiment with more advanced architectures, such as implementing **Transfer Learning** using established models (VGG16, ResNet, etc.) to leverage features pre-learned from massive image datasets.
