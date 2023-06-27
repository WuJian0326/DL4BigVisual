# Deep Learning for Big Visual Data

This repository contains homework assignments for the course "Deep Learning for Big Visual Data." Each homework focuses on different topics and involves coding tasks and questions to answer.

## HW1

### Practices of Python Numpy

### Polynomial Regression

### Classification on Digital Images Using Traditional Machine Learning

Perform image classification on digital images using traditional machine learning methods. This part includes the following functionalities:

- Implement Principal Component Analysis (PCA) to reduce the dimensionality of digital images.
- Use Support Vector Machines (SVM) classifier from scikit-learn for digit classification.
- Evaluate the classification accuracy of the model on the validation set.

## HW2

### Practices of PyTorch

### Digital Image Classification by MLP and CNN

Implement MLP (Multilayer Perceptron) and CNN (Convolutional Neural Network) for digital image classification. It includes the following tasks:

- Prepare the MNIST dataset and preprocess it, including downsampling and splitting into training, validation, and test sets.
- Define an MLP model using PyTorch and apply it to the MNIST dataset for image classification.
- Build a CNN model (LeNet-5) using PyTorch and apply it to the MNIST dataset for image classification.

### Image Classification by CNN

Perform image classification using CNN on the CIFAR-10 dataset. This task includes the following:

- Load and visualize the CIFAR-10 dataset, perform data normalization and augmentation.
- Define a CNN model for image classification on the CIFAR-10 dataset.
- Experiment with different network parameters/configurations and evaluate accuracy, including the number of convolutional layers, kernel size, stride, dilation, and Dropout in fully connected layers.


## HW3

### Image Classification on Imbalanced Dataset

Handle image classification on an imbalanced dataset. Train and evaluate a Convolutional Neural Network (CNN) model on the balanced and imbalanced CIFAR-50-LT dataset. Implement techniques such as resampling, reweighting, and mixing to improve the performance on imbalanced data.

### Transfer Learning and Domain Adaptation

Study domain adaptation using the MNIST and MNIST-M datasets. Train and test models using the DABP (Unsupervised Domain Adaptation by Backpropagation) method to transfer knowledge from the source domain to the target domain.

### GAN (Generative Adversarial Network)



## HW4

### Problem 1: Faster RCNN (20%)

This problem involves using Detectron2, a software system for object detection, to perform various tasks such as running inference on pre-trained models, training custom models on specific datasets (traffic sign and balloon), and evaluating the performance of the models.

### Problem 2: Tracktor for Pedestrian Multi-Object Tracking (20%)

In this problem, you will explore Tracktor, a multiple object tracking method that utilizes object detection and appearance embedding models. You will train an object detector on the MOT-17 dataset and then run the Tracktor algorithm for multi-person tracking. Evaluation of the tracking performance is also required.

### Problem 3: Mask R-CNN on a Custom Dataset (10%)

This problem focuses on training an instance segmentation model using Mask R-CNN. While the traffic sign dataset used in Problem 1 only contains bounding box labels, here you will switch to the balloon segmentation dataset, which includes segmentation mask information. You will fine-tune a pre-trained Mask R-CNN model on the balloon dataset and evaluate its performance using the COCO API.

### Problem 4: 2D Human Pose Estimation (25%)

The final problem involves human pose estimation using the MPII dataset and the Stacked Hourglass Network. You will train the network on the MPII dataset, evaluate it on the validation set, perform inference using pretrained models, and visualize the human pose estimation results.


## HW5

### Problem 1: PointNet (30%)

Implement PointNet for point cloud object classification. Solve questions related to PointNet model implementation, training, and inference visualization.

### Problem 2: Vision Transformer for Image Classification (45%)

Perform inference with the Vision Transformer (ViT) model for image classification. Answer questions related to using different ViT models, the shape of [CLS] logits, and adapting pre-trained ViT models to new datasets like CIFAR-10.

### Task B: Fine-tune the Vision Transformer on CIFAR-10

Fine-tune a pre-trained Vision Transformer model on the CIFAR-10 dataset. Answer questions related to fine-tuning with end-to-end and fixed backbone settings and their differences.

## Usage

Each assignment or project has its own folder containing the necessary code and any accompanying resources. Navigate to the respective folder to access the specific files and follow the instructions provided to run the code or reproduce the results.
