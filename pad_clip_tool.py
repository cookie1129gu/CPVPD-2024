import os
import numpy as np
from skimage import io
import cv2


def process_raster_images(input_dir, output_dir, slice_size=256):
    """
    Process raster images by padding them to be divisible by slice_size and splitting into slices of slice_size.

    Args:
        input_dir (str): Path to the input folder containing raster images.
        output_dir (str): Path to the output folder where slices will be saved.
        slice_size (int): Size of each slice, default is 256.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all raster images in the input folder
    for filename in os.listdir(input_dir):
        if not filename.endswith(('.result', '.tiff')):
            continue

        file_path = os.path.join(input_dir, filename)
        img = io.imread(file_path)  # Read the image

        # Get original image size
        original_size = img.shape[:2]  # (height, width)
        print(f"Processing {filename} - Original size: {original_size}")

        # Calculate required padding
        pad_height = (slice_size - (original_size[0] % slice_size)) % slice_size
        pad_width = (slice_size - (original_size[1] % slice_size)) % slice_size
        padding = ((0, pad_height), (0, pad_width), (0, 0))  # H, W, C
        img_padded = np.pad(img, padding, mode='constant', constant_values=255)  # Pad with white

        # Record padded size
        padded_size = img_padded.shape[:2]
        print(f"Padded size: {padded_size} (Added height: {pad_height}, width: {pad_width})")

        # Create folder named after the original file
        file_output_dir = os.path.join(output_dir, filename[:-4])
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        # Save padding information to text file
        padding_info_path = os.path.join(file_output_dir, "padding_info.txt")
        with open(padding_info_path, 'w') as f:
            f.write(f"Original size: {original_size}\n")
            f.write(f"Padded size: {padded_size}\n")
            f.write(f"Padding added - Height: {pad_height}, Width: {pad_width}\n")

        # Split image into 256x256 slices
        num_slices_height = padded_size[0] // slice_size
        num_slices_width = padded_size[1] // slice_size

        for i in range(num_slices_height):
            for j in range(num_slices_width):
                slice_img = img_padded[
                            i * slice_size:(i + 1) * slice_size,
                            j * slice_size:(j + 1) * slice_size
                            ]

                # Save the slice
                slice_filename = f"{i}_{j}.result"
                slice_path = os.path.join(file_output_dir, slice_filename)
                io.imsave(slice_path, slice_img)

    print(f"All images processed and saved to {output_dir}")
