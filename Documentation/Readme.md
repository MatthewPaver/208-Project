
# Architexa Project Documentation

## Table of Contents

1. [Access Instructions](#access-instructions)
2. [User Manual](#user-manual)
3. [Coding Documentation](#coding-documentation)
4. [Design Documentation](#design-documentation)
5. [Test Documentation](#test-documentation)
6. [Technical Feasibility Study](#technical-feasibility-study)
7. [Market Research](#market-research)
8. [Model Factsheet](#model-factsheet)

---

## Access Instructions

Detailed steps to access the backend and repository:

1. Open the following GitHub link to access our repository: [Architexa Repository](https://github.com/Ash237333/208-Project).
2. The repository includes several folders:
   - **Legacy Code:** Contains old versions of scripts and codebases used during development.
   - **Dataset:** Contains all code related to creating the dataset, along with the actual dataset and an unaugmented version.
   - **Test Trained Model:** Includes a script (`test_model.py`) to run our trained model and generate an output image (`test.png`).
   - **Neural Network Code:** Contains all code related to building, training, and tuning the model. The `worker.py` script can be run to train the model.
3. Each folder includes a `README.md` file for additional guidance.

## User Manual

Guide for common users to generate images:

1. Navigate to our website: [Architexa](https://architexa-1f4ce.web.app/).
2. Sign up and create an account or log in if you already have one.
3. Enter a short sentence describing the image you want to generate and click submit.
4. Wait for about 75 seconds for the generation process. The UI will display four generated images matching your prompt.

## Coding Documentation

Detailed descriptions of scripts and their functionalities:

- **Image Generator - Pinterest Version:** A Python script to automate scraping and downloading images from Pinterest.
- **Image Generator - BulkAI Version:** Automates the process of generating images based on user prompts, includes functionalities like downloading, upscaling, and variation.
- **Image Pre-Processing Script:** Prepares datasets for training, testing, and validation by resizing, augmenting, normalizing, and adding noise to images.
- **Model Development:** Scripts for building the generator and discriminator components of the neural network, training loops, and hyperparameter tuning.

## Design Documentation

High-level architectural design and implementation details:

- **Generative Models:** Learn the probability distribution of features in a dataset to generate images.
- **GAN Architecture:** Trains on random noise and outputs perturbed versions of the input, mimicking the distribution of real data.
- **Prompt-Based AI Image Generators:** Assign keyword labels to images and train the model to generate images based on these keywords.

## Test Documentation

Results and outcomes of various tests conducted on the neural network and website:

- **Neural Network Tests:** Validate the network's ability to learn features, differentiate classes, and generate human-recognizable images.
- **Dataset Tests:** Ensure legal use of images, consistency in image size, and efficient data transmission.
- **Optimization Tool Tests:** Validate functionality and performance of optimization tools in training and improving neural networks.
- **Website Tests:** Confirm accessibility, functionality, and performance of the website and its integration with the neural network.

## Technical Feasibility Study

Analysis of the technical aspects and feasibility of the project:

- **High-Level Description:** Generative models learn the probability distribution of features in a dataset and generate images with the most likely set of features.
- **Training Inputs:** Random data is used to train the generator, aiming to fit the distribution of real data.
- **Architecture:** Includes convolutional layers, batch normalization, and ReLU layers to upscale and transform the input noise into realistic images.

## Market Research

Insights into the market potential and application of AI-generated imagery:

- **AI Image Generation Trends:** Platforms like DALL-E and MidJourney highlight the growing popularity and potential of AI-generated content.
- **BinaryGAN:** A specialized GAN for architectural design, offering efficient training and focused image generation capabilities.
- **Integration with Architectural Software:** Potential for enhancing design workflows and improving the accuracy and detail of generated outputs.

## Model Factsheet

Detailed information on the types and architectures of GANs used:

- **GAN (Generative Adversarial Network):** Consists of a generator and a discriminator, where the generator creates images and the discriminator tries to distinguish between real and fake images.
- **DCGAN (Deep Convolutional GAN):** Uses convolutional-transpose layers to turn random input into images, with batch normalization and ReLU layers for stabilization.
- **Binary GAN:** A type of DCGAN that works with binary data, offering faster performance and precision.
- **cGAN (Conditional GAN):** Extends DCGANs by using labeled datasets to generate images with specific attributes.
- **Image-to-Image Generator:** Uses a U-Net architecture for tasks like translating sketches into detailed images.

---

For more detailed information, please refer to the individual documents provided in the repository.
