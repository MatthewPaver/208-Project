# Image Pre-Processing Script

This Python script is designed for pre-processing images for machine learning tasks, particularly for preparing datasets for training, testing, and validation in a Convolutional Neural Network (CNN) or other image-based machine learning models. The script includes functionalities such as resizing, augmenting, normalising, and adding noise to images, followed by splitting the dataset into training, testing, and validation sets.

## Features

- **Resizing**: Changes the dimensions of the images to 1024x1024 pixels to ensure uniformity.
- **Augmentation**: Applies grayscale conversion, rotation, and horizontal flipping to images to enhance the robustness of the model.
- **Normalisation**: Adjusts pixel values to a range of [0, 1] for better model convergence.
- **Noise Addition**: Introduces random noise to images to reduce overfitting and improve generalization.
- **Dataset Splitting**: Divides the processed images into training, testing, and validation sets based on standard splitting ratios.

## Setup

Before running the script, ensure you have Python installed on your system and the following libraries are installed:

- Pillow
- NumPy
- scikit-learn

You can install these libraries using pip:

```bash
pip install Pillow numpy scikit-learn

## Usage
## Prepare Your Dataset: Organize your images into subfolders within a root directory. Each subfolder should represent a category or class of images.

## Configure the Script:

- **Set** the root_directory variable to the path of your root dataset folder.
- **Set** the processed_images_directory variable to the path where you want the processed images to be saved.
- **Run** the Script: Execute the script using Python.

python image_preprocessing.py
