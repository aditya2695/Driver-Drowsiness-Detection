# Driver Drowsiness Detection using Deep Learning


This project aims to detect drowsiness using computer vision techniques. It utilizes PyTorch, OpenCV, and Haar Cascade for face detection. Ensemble learning is implemented using AlexNet and VGG16 models to enhance accuracy and performance.

## Dataset

<img src="images/data_sample.png" width="500">

## Features





## Features

    Drowsiness detection using computer vision
    Face detection using Haar Cascade classifier
    Ensemble learning with AlexNet and VGG16 models
    PyTorch for deep learning implementation
    OpenCV for image processing and manipulation

## Ensemble Learning

Ensemble learning is employed to combine the predictions from multiple models to improve the accuracy and robustness of the drowsiness detection system. In this project, we utilize the AlexNet and VGG16 models for ensemble learning.

**AlexNet**: AlexNet is a deep convolutional neural network architecture that gained prominence by winning the ImageNet Large Scale Visual Recognition Challenge in 2012. It consists of eight layers, including five convolutional layers and three fully connected layers. By incorporating AlexNet into the ensemble, we leverage its ability to extract hierarchical features from input images.

**VGG16**: VGG16 is another widely used convolutional neural network architecture that achieved high accuracy on the ImageNet challenge. It has 16 layers, including 13 convolutional layers and three fully connected layers. VGG16 is known for its simplicity and effectiveness in capturing complex image patterns.


The predictions from each model are averaged to obtain the final prediction. By combining the outputs of multiple models, we aim to leverage the diverse representations and learn more comprehensive patterns for accurate drowsiness detection.