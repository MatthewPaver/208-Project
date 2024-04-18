import yaml
import os

# Ask for user input
album_name = input("What would you like to call the album? ")
prompt_base = input("What prompt do you want? (E.g., Photo realistic image of the front of a brick house) ")
num_images = int(input("How many images do you want? "))

# Generate prompts based on the number of images
prompts = [f"{prompt_base}-{i+1}" for i in range(num_images)]

# Configuration dictionary
config = {
    "bot": "midjourney",
    "album": album_name,
    "download": True,
    "upscale": True,
    "variation": True,  # Set based on your preference
    "thumbnail": False,
    "suffix": " --ar 3:2",
    "prompt": prompts,
    "wait": "180s"  # Adds a 20-second wait between prompts
}

# Define the YAML file path
yaml_file_path = os.path.expanduser("~/INSERT FILE PATH HERE/bulkai.yaml")

# Save the configuration to a YAML file
with open(yaml_file_path, "w") as file:
    yaml.dump(config, file, default_flow_style=False)

print("YAML configuration file has been generated at:", yaml_file_path)
