import os
import random
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def resize_image(image, size=(1024, 1024)):
    """Resize the image to the specified size."""
    return image.resize(size, Image.LANCZOS)

def augment_image(image):
    """Apply augmentation techniques: grayscale conversion, rotation, and flipping."""
    # Convert to grayscale
    if random.choice([True, False]):
        image = ImageOps.grayscale(image)
    
    # Rotate the image by a random angle between -25 and 25 degrees
    if random.choice([True, False]):
        image = image.rotate(random.uniform(-25, 25))
    
    # Flip the image horizontally
    if random.choice([True, False]):
        image = ImageOps.mirror(image)
    
    return image

def normalise_image(np_image):
    """Normalize image pixel values to the range [0, 1]."""
    return np_image.astype('float32') / 255.0

def add_noise(np_image, noise_factor=0.05):
    """Add random noise to the image."""
    noise = np.random.randn(*np_image.shape) * noise_factor
    np_image_noisy = np_image + noise
    return np.clip(np_image_noisy, 0., 1.)

def process_images(image_dir):
    """Process all images in the specified directory."""
    print(f"Processing images in {image_dir}")
    processed_images = []
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                with Image.open(os.path.join(root, filename)) as img:
                    img = resize_image(img)
                    img = augment_image(img)
                    np_img = np.array(img)
                    np_img = normalise_image(np_img)
                    np_img = add_noise(np_img)
                    processed_images.append((np_img, filename))
    print(f"Finished processing images in {image_dir}")
    return processed_images

def split_and_save_images(images, output_dir, folder_path):
    """Split images into train and test sets and save."""
    category = os.path.basename(folder_path)
    if not images:
        print(f"No images found for {category}")
        return
    print(f"Splitting and saving images for {category}")
    train, test = train_test_split(images, test_size=0.4, random_state=42)

    for dataset, name in [(train, 'train'), (test, 'test')]:
        folder_path = os.path.join(output_dir, name, category)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for img, filename in dataset:
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(folder_path, filename))
    print(f"Finished splitting and saving images for {category}")

def process_folders(root_dir, processed_images_directory):
    """Process each subdirectory within the root directory."""
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            folder_path = os.path.join(root, dir)
            print(f"Processing and splitting folder: {folder_path}")
            images = process_images(folder_path)
            split_and_save_images(images, processed_images_directory, folder_path)

root_dir = r'C:\\Users\\INSERT DIRECTORY HERE'
processed_images_directory = r'C:\\Users\\INSERT DIRECTORY HERE'

# Call the function to start processing
process_folders(root_dir, processed_images_directory)
