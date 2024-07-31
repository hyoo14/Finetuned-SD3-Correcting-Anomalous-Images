import os
from PIL import Image

def convert_images_in_directory_renamed(image_directory, output_directory, target_format='JPEG'):
    # Check if the output directory exists, create if not
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # List all files in the directory containing 'DALL' and ending with '.webp'
    image_files = [f for f in os.listdir(image_directory) if 'DALL' in f and f.endswith('.webp')]

    # Process each image with a counter for renaming
    counter = 1
    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        try:
            # Open and convert the image
            image = Image.open(image_path)
            converted_image = image.convert('RGB')  # Converting to 'RGB' as a general format

            # Create the output file path with a sequential name
            output_file_path = os.path.join(output_directory, f"{counter:02d}.jpeg")

            # Save the converted image
            converted_image.save(output_file_path, format=target_format)
            print(f"Converted {image_file} to {output_file_path}")

            # Increment the counter
            counter += 1

        except Exception as e:
            print(f"Failed to convert {image_file}: {e}")

### usage ###
## Define directories
#image_directory = "/content"
#output_directory = "/content2"

## Convert images
#convert_images_in_directory_renamed(image_directory, output_directory)
