# README

This README provides a comprehensive guide on how to run the `bulkai generate` command in Terminal using your YAML configuration file and gives an overview of the `Generate Images.py` script used to automate the image generation process.

## Getting Started

### Setting Up Your Environment

1. **Open Terminal.**

2. **Run the Go Environment Path Command**

    Add `go` binaries to your path by executing the following command in the terminal:

    ```shell
    export PATH=$PATH:$(go env GOPATH)/bin
    ```

3. **Navigate to Your YAML File Directory**

    Change the current directory to where your YAML file is located:

    ```shell
    cd /Users/INSERT DIRECTORY HERE
    ```

### Creating a Session

4. **Create a Session with BulkAI**

    Use the `bulkai create-session` command to initiate a session file. This will open a Chrome window where you will need to log in to Discord.

    ```shell
    bulkai create-session
    ```

### Generating Files

5. **Run the Generate Command**

    Start the generation process with the following command:

    ```shell
    bulkai generate --config bulkai.yaml
    ```

    You can stop the generation process anytime using `Ctrl+C` and resume later by rerunning the same command.

   For more detailed information and the latest updates, please visit the [BulkAI GitHub page](https://github.com/bulkai).
   

## Generate Images.py

The `Generate Images.py` script automates the process of generating images. Below is a detailed breakdown of its functionality:

### Input Collection
- **Album Name**: Users are prompted to enter a name for the image album.
- **Image Prompt**: Users provide a base prompt which is used to define the theme or concept for the images.
- **Number of Images**: Users specify how many images they want to generate.

### Prompt Construction
- The script constructs a list of unique prompts for each image by appending an index to the base prompt.

### Configuration and Output
- **Configuration**: All user inputs and settings are compiled into a configuration dictionary.
- **YAML Configuration**: The configuration is saved to a YAML file, allowing for repeated use or modification.
- **Bot Compatibility**: The script is tailored to interact with the `midjourney` bot.
- **Options**:
  - **Download**: Enables downloading of generated images.
  - **Upscale**: Allows for the enhancement of image resolution.
  - **Variation**: Offers variation in image generation.
  - **Thumbnail**: Option to generate thumbnails is available but disabled by default.
  - **Suffix**: Custom suffixes can be added to commands as needed.
  - **Wait Time**: Configurable wait time between issuing prompts to manage API calls or bot interactions.

The YAML configuration file location is set by default to `~INSERT FILE PATH/bulkai.yaml`, and users are notified of the file path upon successful creation.
