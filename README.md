##  UNet with ResNet Backbone: A Dynamic Fusion for Semantic Segmentation
In the realm of semantic segmentation, where pixel-level understanding of images is paramount, the synergy between UNet and ResNet emerges as a compelling solution. UNet, recognized for its intricate localization abilities, and ResNet, celebrated for its depth and feature extraction prowess, come together in a fusion that amplifies their individual strengths. This brief introduction sets the stage for exploring the amalgamation of UNet with ResNet as its backbone, unveiling its capacity to excel across diverse segmentation tasks while retaining architectural elegance and efficiency.

### Dataset
_The Sandstone dataset is a benchmark dataset used in image segmentation tasks, particularly in geological image analysis. It contains high-resolution images of sandstone rock samples, often obtained through microscopy techniques. Researchers use this dataset to develop and evaluate algorithms for segmenting different components within the sandstone samples, aiding in the understanding of their microstructure and properties._

This Dataset is annotated into 4 different classes :
1. Bentheim (B49): Known for its uniformity and fine-grained structure, Bentheim sandstone is commonly used in construction and architectural applications due to its durability and aesthetic appeal.

2. Berea (Br46): Berea sandstone is characterized by its high porosity and permeability, making it valuable for research in petroleum engineering and hydrology, as well as for geological studies.

3. Fontainebleau (F57): Fontainebleau sandstone is renowned for its tight grain structure and exceptional strength, making it a popular choice for rock climbing holds and as a reference material in laboratory experiments.

4. Gildehausen (G44): Gildehausen sandstone is notable for its heterogeneous composition, often exhibiting variations in grain size and mineral content, making it an interesting subject for geological research and analysis.

This Dataset originally contain 2 Files : 
1. images.tif
2. masks.tif

Each being a **_"tiff stack file"_** having 1600 slices and dimension being _128x128_, all the individual slices are extracted into 1600 _single .tif files_ each having dimension : _128x128_.

![images_0](https://github.com/arpsn123/Mitocondria-Segmentation/assets/112195431/26ed1f4b-a73f-4135-8f7b-24042b9a60fe)
![masks](https://github.com/arpsn123/Mitocondria-Segmentation/assets/112195431/c0d406cf-0bcb-4bb6-a7d6-6fa6dd16ee50)

Imgaes used for Training the model : 90% --> 1,440 images.

Imgaes used for Testing the model: 10% --> 160 images.

### Model
Here, the [segmentation_models](https://github.com/qubvel/segmentation_models.git) library in Python, used extensively this for **_2D image segmentation task_**. This module offers a streamlined approach to implementing the _Pretrained Machine Learning Models_, with the **UNet model enhanced by ResNet** serving as its backbone.
