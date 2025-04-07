# CIFAR-10 Image Classifier using MobileNetV2

This project implements an image classifier for the CIFAR-10 dataset using **TensorFlow**, **Keras**, and **MobileNetV2** as a pre-trained feature extractor. It resizes CIFAR-10 images, augments a deep learning model with custom layers, and trains the model to achieve high accuracy.

## 📦 Dataset

We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

Classes include:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## 🚀 Model Architecture

We use **MobileNetV2** (pre-trained on ImageNet) as the base model for feature extraction. The final classification layers are custom and trained on CIFAR-10:

- `GlobalMaxPooling2D`
- Dense layers with ReLU activations and L2 regularization
- Dropout layers for regularization
- Final `Dense` layer with `softmax` for 10-class classification

## 🧪 Training Setup

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Epochs: 20
- Early stopping based on training loss
- Checkpointing to save model weights

## 🧠 Training & Evaluation

- CIFAR-10 images are resized from `32x32` to `96x96` to meet the input requirements of MobileNetV2.
- The training process includes real-time monitoring of loss and early stopping if performance stagnates.
- The model is evaluated on the test dataset and aims to achieve >85% accuracy.

## 📊 Training Loss Plot

After training, the loss per epoch is plotted using `matplotlib`.

## 🔧 Notes

- MobileNetV2 layers are partially frozen for faster training.
- You can fine-tune more layers or experiment with different architectures for improved performance.
