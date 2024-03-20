import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

def resize_image(image, size=(1024, 1024)):
    """Resize the image to the specified size."""
    return image.resize(size, Image.ANTIALIAS)

def augment_image(image):
    """Apply augmentation techniques: grayscale conversion, rotation, and flipping."""
    image = ImageOps.grayscale(image)  # Convert to grayscale
    image = image.rotate(15)  # Rotate the image
    image = ImageOps.mirror(image)  # Flip the image horizontally
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
    processed_images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(os.path.join(image_dir, filename)) as img:
                img = resize_image(img)
                img = augment_image(img)
                np_img = np.array(img)
                np_img = normalize_image(np_img)
                np_img = add_noise(np_img)
                processed_images.append((np_img, filename))
    return processed_images

def split_and_save_images(images, output_dir, category):
    """Split images into train, test, validate sets and save."""
    train, test = train_test_split(images, test_size=0.2, random_state=42)
    train, validate = train_test_split(train, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    for dataset, name in [(train, 'train'), (test, 'test'), (validate, 'validate')]:
        folder_path = os.path.join(output_dir, name, category)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for img, filename in dataset:
            Image.fromarray((img * 255).astype(np.uint8)).save(os.path.join(folder_path, filename))

def process_folders(root_dir, output_dir):
    """Process each subdirectory within the root directory."""
    for category in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, category)
        if os.path.isdir(folder_path):
            print(f"Processing and splitting folder: {category}")
            images = process_images(folder_path)
            split_and_save_images(images, output_dir, category)

#  usage
root_directory = 'path/to/dataset'
processed_images_directory = 'path/to/processed_images'
process_folders(root_directory, processed_images_directory)
