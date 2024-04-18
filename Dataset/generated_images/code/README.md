# README

This README provides a comprehensive guide on how to run the `bulkai` generate command in Terminal using your YAML configuration file.

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
