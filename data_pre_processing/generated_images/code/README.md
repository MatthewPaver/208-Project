# README

This README provides a comprehensive guide on how to run the `bulkai` generate command in Terminal using your YAML configuration file, and how to install and use `labelImg`.

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
    cd /Users/mattpaver/Desktop/Generated Images/Code
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

## LabelImg: Image Annotation Tool

`labelImg` is a versatile tool for annotating images, making it an essential utility for tasks that require precise object labeling, such as training machine learning models. Below you'll find detailed instructions on how to install and effectively use `labelImg`.

### Installation

`labelImg` can be easily installed via `pip`, Python's package installer. Ensure you have Python and `pip` installed on your system before proceeding. To install `labelImg`, open your terminal and execute the following command:

```shell
pip install labelImg
```

This command downloads and installs `labelImg` and its dependencies. If you encounter any permission issues, you might need to add `sudo` at the beginning of the command (for macOS/Linux) or run the command prompt as an administrator (for Windows).

### Launching labelImg

After installation, you can start `labelImg` from the terminal. Navigate to the directory containing the images you wish to annotate and run:

```
labelImg
```

This command opens the `labelImg` graphical user interface, ready for you to begin annotating your images.

### Using labelImg

Here's a step-by-step guide to using `labelImg` for image annotation:

- **Open Directory**: In the `labelImg` interface, click on 'Open Dir' to select the directory containing your images.
- **Change Save Directory (Optional)**: If you wish to save your annotations in a specific directory, click on 'Change Save Dir' to specify the location.
- **Create RectBox**: Click on 'Create RectBox' or press `W` to start drawing a bounding box around the object you want to label.
- **Label**: After drawing the box, a dialog will prompt you to enter the label for the object. Type the label and press enter.
- **Next Image**: Navigate through your images by clicking 'Next Image' or pressing `D`. To go back, click 'Prev Image' or press `A`.
- **Save**: Your annotations are automatically saved in XML format in the Pascal VOC format when you move to the next image. You can also manually save by clicking 'Save' or pressing `Ctrl+S`.

### Tips for Efficient Annotation

- **Shortcuts**: Familiarize yourself with `labelImg` shortcuts to speed up your annotation process. Key shortcuts include `W` (Create RectBox), `D` (Next Image), `A` (Previous Image), and `Ctrl+S` (Save).
- **Predefined Classes**: To streamline the labeling process, you can preload a set of predefined classes. Create a text file with one class name per line and load it via 'Open/Open Annotation Classes File'.