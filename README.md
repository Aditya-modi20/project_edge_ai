# Project_edge_ai
The main task of this project is to build an android application that can predict a person age, gender and emotion from the input face image. 

## Methodology 
Three different models are trained to predict age, gender and emotion of a person from the input image. 
The gender and emotion model are trained on sequential model and the age model is trained on Resnet-50 due to its complexity. 

## Dataset used 
- Age dataset **[link](https://www.kaggle.com/datasets/himanshuydv11/facial-emotion-dataset)**
- Gender dataset **[link](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)**
- Emotion dataset **[link](https://www.kaggle.com/datasets/alifshahariar/utkface-dataset-face-aligned-and-labeled)**

## Model architecture
- Real time age, gender and emotion detection on edge devices.
- Recognize 
  -- Age group (7 classes)
  -- Gender (2 classes)
  -- Emotion (6 classes)

## Link

Link to google drive **[Link](https://drive.google.com/drive/folders/1aASTu2Zq8hBGnqQuwQ2ddpfrXdZt4ekM?usp=sharing)**

## Permission
The app requires permission to access the internal storage to access the device's gallery.

## Usage 
1. Open the application on the android device.
2. Grant the necesssary permissions. 
3. Click on "capture" button to capture an image or click "gallery" button to upload a image. 
4. After selecting the app will process the image using the tflite models.
5. The final result will be displayed in the application.
