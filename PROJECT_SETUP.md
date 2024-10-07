# AccessEye Project Setup

## Gathering Data and Choosing a Model Architecture
To build a dataset, we retrieved exisiting datasets featuring images of vehicles, license plates, and the International Symbol of Access. We then annotated the images to classify them into the following classes:
- **vehicle**
- **license_plate**
- **symbol_of_access**

We chose **Roboflow 3.0 Object Detection (Fast)** as the model type, utilizing the **COCOn** checkpoint to leverage pre-trained models, accelerating the initial learning phase and increasing detection accuracy.

## Generating our Custom Dataset
To maximize our model, we applied the following preprocessing and augmentation techniques using Roboflow:
- **Pre-Process:**
  - Auto-Orient: Adjusts image orientation automatically
  - Resize: Standardizes images to 640x640 pixels
- **Data Augmentation:**
  - Crop: Applies random cropping to images, ranging from 0% to 20% zoom
  - Rotation: Randomly rotates images between -15° and +15° to simulate different angles
  - Shear: ±10° horizontally and vertically
  - Noise: Adds noise to up to 0.1% of the pixels
- **Split Data:**
  - 87% training, 8% validation, 4% testing

## Downloading Custom Data
Utilized the YOLOv7 PyTorch export feature from Roboflow to obtain our custom annotated dataset and used a Jupyter notebook snippet to integrate the dataset into our Google Colab workspace.
```python
!git clone https://github.com/WongKinYiu/yolov7
%cd yolov7
!pip install -r requirements.txt
```
```python
!pip install roboflow
%cd yolov7

from roboflow import Roboflow
rf = Roboflow(api_key="Zi2LVnTfSriJOC0z7Iz9")
project = rf.workspace("hacksmu-nhjks").project("dataset-hacksmu")
version = project.version(1)
dataset = version.download("yolov7")
```
## Custom Training
**Initial Checks:** Conducted a preliminary training run for 1 epoch to validate the setup before extending the process to a full training regimen.
```python
%cd /content/yolov7
!wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt
```
```python
%cd /content/yolov7
!python train.py --batch 16 --cfg cfg/training/yolov7.yaml --epochs 1 --data {dataset.location}/data.yaml --weights 'yolov7_training.pt' --device 0
```
![image](https://github.com/user-attachments/assets/334a5fa5-d39d-40d6-92a2-979d0e8c2f54)
## Evaluation
Once the model was completed, we assessed the model’s accuracy and efficiency on a set of test images to ensure reliable real-time detection.
```python
!python detect.py --weights runs/train/exp/weights/best.pt --conf 0.1 --source {dataset.location}/test/images

import glob
from IPython.display import Image, display

i = 0
limit = 10000 # max images to print
for imageName in glob.glob('/content/yolov7/runs/detect/exp2/*.jpg'): #assuming JPG
    if i < limit:
      display(Image(filename=imageName))
      print("\n")
    i = i + 1
```
_Here are some example inferences displayed on test images:_

![Untitled presentation](https://github.com/user-attachments/assets/383a5d0c-1f50-429f-97b0-ca47fb8be5fa)

## Deployment
After thorough training and validation, the model's learned parameters were encapsulated into exportable weights. These weights were prepared for deployment to integrate the detection system into operational environments where real-time analysis is critical.
``` python
!zip -r export.zip runs/detect
!zip -r export.zip runs/train/exp/weights/best.pt
!zip export.zip runs/train/exp/*
```
