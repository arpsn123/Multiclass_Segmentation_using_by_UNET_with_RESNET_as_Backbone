# UNet with ResNet Backbone: A Dynamic Fusion for Semantic Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.7.0-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.7.0-green.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21.0-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.3-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4.3-purple.svg)
![segmentation_models](https://img.shields.io/badge/Segmentation%20Models-0.2.1-lightblue.svg)

## Overview

In the realm of semantic segmentation, where pixel-level understanding of images is paramount, the synergy between **UNet** and **ResNet** emerges as a compelling solution. UNet, recognized for its intricate localization abilities, excels in tasks requiring precise delineation of objects within images. In contrast, ResNet, celebrated for its depth and feature extraction prowess, brings robustness to the model through its innovative residual connections. Together, this fusion amplifies their individual strengths, creating a framework that is both architecturally elegant and highly efficient. This document sets the stage for exploring the amalgamation of UNet with ResNet as its backbone, unveiling its capacity to excel across diverse segmentation tasks while maintaining operational effectiveness.

## Dataset

The **Sandstone dataset** is a benchmark dataset utilized in image segmentation tasks, particularly in the analysis of geological images. This dataset comprises high-resolution images of sandstone rock samples, often captured through advanced microscopy techniques. Researchers leverage this dataset to develop and evaluate algorithms for segmenting different components within the sandstone samples, facilitating a deeper understanding of their microstructure and properties.

### Class Annotations

The dataset is meticulously annotated into four distinct classes:

1. **Bentheim (B49)**: Known for its uniformity and fine-grained structure, Bentheim sandstone is commonly employed in construction and architectural applications due to its durability and aesthetic appeal.

2. **Berea (Br46)**: Characterized by its high porosity and permeability, Berea sandstone is valuable for research in petroleum engineering and hydrology, as well as for geological studies.

3. **Fontainebleau (F57)**: Renowned for its tight grain structure and exceptional strength, Fontainebleau sandstone is a popular choice for rock climbing holds and serves as a reference material in laboratory experiments.

4. **Gildehausen (G44)**: Notable for its heterogeneous composition, Gildehausen sandstone often exhibits variations in grain size and mineral content, making it an interesting subject for geological research and analysis.

### Dataset Structure

The dataset originally contains two files:

- **images.tif**
- **masks.tif**

Each of these files is a **"tiff stack file"** comprising 1600 slices with dimensions of _128x128_. All individual slices are extracted into 1600 single .tif files, each maintaining the dimensions of _128x128_ pixels.

![Images Sample](https://github.com/arpsn123/Mitocondria-Segmentation/assets/112195431/26ed1f4b-a73f-4135-8f7b-24042b9a60fe)
![Masks Sample](https://github.com/arpsn123/Mitocondria-Segmentation/assets/112195431/c0d406cf-0bcb-4bb6-a7d6-6fa6dd16ee50)

- **Images used for Training the model**: 90% (1,440 images)
- **Images used for Testing the model**: 10% (160 images)

### Data Preprocessing

Data preprocessing is a crucial step in the pipeline, ensuring that the input data is normalized and augmented effectively. This enhances the model's ability to generalize from the training data.

```python
import cv2
import numpy as np
import os

def load_and_preprocess_images(image_dir, mask_dir):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.tif'):
            img = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            
            mask = cv2.imread(os.path.join(mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (128, 128))
            masks.append(mask)
    
    return np.array(images), np.array(masks)

images, masks = load_and_preprocess_images('path/to/images', 'path/to/masks')
```

### Model Architecture

In this project, the [segmentation_models](https://github.com/qubvel/segmentation_models.git) library in Python is extensively employed for **2D image segmentation tasks**. This library offers a streamlined approach to implementing pretrained machine learning models, enabling rapid development and experimentation.

The **UNet model**, enhanced by the ResNet backbone, serves as a robust architecture for segmentation tasks. The architecture is built using the Keras API as follows:

```
python
from segmentation_models import UNet
from keras import Input

# Define input layer
input_layer = Input(shape=(128, 128, 1))

# Create UNet model with ResNet backbone
model = UNet(backbone_name='resnet34', input_shape=(128, 128, 1), classes=4, activation='softmax')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

```

### Training the Model

Model training involves feeding the preprocessed images and masks into the network. It is crucial to monitor the training process to avoid overfitting. 

```python
from keras.callbacks import EarlyStopping

# Set early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(images, masks, validation_split=0.1, epochs=50, batch_size=16, callbacks=[early_stopping])
```

### Performance Evaluation

Evaluating the model’s performance is essential to ensure its efficacy in segmentation tasks. Metrics such as Intersection over Union (IoU) and pixel accuracy are commonly used in semantic segmentation.

```python
from sklearn.metrics import jaccard_score

# Function to calculate IoU
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)

# Evaluate model on test data
test_predictions = model.predict(test_images)
iou_scores = [calculate_iou(test_masks[i], test_predictions[i]) for i in range(len(test_masks))]
print(f'Mean IoU: {np.mean(iou_scores)}')
```

### Conclusion

Through this integrated approach, the model not only benefits from the localization capabilities of UNet but also leverages the deep feature representation provided by ResNet, making it particularly effective for challenging image segmentation tasks in geological analysis. This project showcases the power of combining advanced neural network architectures to achieve high performance in semantic segmentation.

### Future Work

Further enhancements may include experimenting with different backbones, incorporating advanced data augmentation techniques, and refining the model to improve performance on edge cases within the dataset. Additionally, exploring ensemble methods could provide insights into improving the overall accuracy and robustness of the segmentation outputs.
