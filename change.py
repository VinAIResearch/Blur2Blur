import os
from PIL import Image

def convert_images_to_jpeg(image_folder):
    # Create an output folder if it does not exist
    output_folder = os.path.join(image_folder, 'jpeg_converted')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the image folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # Open an image file
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                # Convert image to RGB mode if it's not already
                img = img.convert('RGB')
                
                # Save the image as JPEG
                base_filename = os.path.splitext(filename)[0]
                jpeg_path = os.path.join(output_folder, f"{base_filename}.jpeg")
                img.save(jpeg_path, 'JPEG')

# Specify the folder containing images
image_folder = 'img'

convert_images_to_jpeg(image_folder)