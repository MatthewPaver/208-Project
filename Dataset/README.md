# Contents

This repository includes various resources and scripts designed for image processing and machine learning tasks:

## Completed Dataset
Contains test and train datasets, which are derived from the pre-processing script. This folder holds the manipulated images ready for use in machine learning models.

## Pinterest Scraper
A Python script that automates downloading images from Pinterest using search terms from subdirectory names. It includes features to eliminate duplicate images via content hashing to ensure a unique image set.

## Pre Processing
A Python script tailored for image pre-processing to suit machine learning models, specifically CNNs. It handles tasks such as resizing, augmenting, normalizing, and adding noise to images, and splits them into training, testing, and validation sets.

## Unaugmented Dataset
Houses images sourced from the `generate_images` script, which are not further processed or augmented.

## generated_images/code

This folder contains scripts and configuration files for automating the image generation process. The main components include:

- **Environment Setup**: Instructions to configure the terminal environment to integrate Go binaries.
- **Session Creation**: Guide to initiate a session using `bulkai create-session`, necessary for image generation.
- **Image Generation**: Steps to run the `bulkai generate` command with a YAML configuration to automate the image creation.

The `Generate Images.py` script assists in generating images by collecting user inputs (like album name and image prompts) and managing configurations through a YAML file located at `INSERT FILE PATH/bulkai.yaml`. It provides options for customization such as image download, upscale, and variation features.

For more comprehensive details and updates, the BulkAI GitHub page can be referenced.

