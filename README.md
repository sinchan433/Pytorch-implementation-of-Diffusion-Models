# Pytorch implementation of Diffusion Models

This project demonstrates a Guided Diffusion Model implemented in PyTorch for generating images. The core idea behind diffusion models is to gradually add noise to data and then learn to reverse this process to generate new data.

## Project Overview

The project implements a U-Net architecture as the denoising model, which is trained to predict the noise added at each step of the diffusion process. Key components include:

- **U-Net Architecture:** A flexible U-Net with options for using ResNet or ConvNeXt blocks, incorporating attention mechanisms and sinusoidal position embeddings to handle timestep information.
- **Beta Schedules:** Different schedules for controlling the noise addition during the diffusion process (linear, cosine, quadratic, sigmoid).
- **Diffusion Process:** Implementation of the forward diffusion process (adding noise) and the reverse process (sampling/denoising).
- **Training:** The model is trained on the Fashion MNIST dataset to learn the denoising process.
- **Visualization:** Tools to visualize the diffusion process and the generated samples.

## Dataset

The model is trained on the Fashion MNIST dataset, which consists of 70,000 grayscale images of Zalando's article images, divided into a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.

Link of the dataset on Kaggle: [https://www.kaggle.com/datasets/zalando-research/fashionmnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## Implementation Details

The implementation utilizes PyTorch and includes custom modules for the U-Net architecture, attention mechanisms, and utility functions for the diffusion process calculations. The training loop minimizes the Huber loss between the predicted noise and the actual noise.
