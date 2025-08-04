import os
import numpy as np
from skimage import io
import cv2


def restore_image_from_slices(input_dir, slice_size=256):
    """
    Restore original image from slices

    Args:
        input_dir (str): Path to the slice folder containing all image slices.
        slice_size (int): Size of each slice, default is 256.
    """
    # Get all slice files in the folder
    slice_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.TIF')])

    if not slice_files:
        raise ValueError(f"No slice files found in folder {input_dir}.")

    # Check number of slices
    print(f"Number of slice files: {len(slice_files)}")

    # Read the first slice to check its size
    first_slice = io.imread(os.path.join(input_dir, slice_files[0]))
    print(f"Slice size: {first_slice.shape}")

    # Determine the number of rows and columns of slices
    num_slices_height = max(int(max(int(slice_file.split('_')[0]) for slice_file in slice_files)) + 1, 1)
    num_slices_width = max(int(max(int(slice_file.split('_')[1].split('.')[0]) for slice_file in slice_files)) + 1, 1)

    print(f"Number of slice rows: {num_slices_height}, columns: {num_slices_width}")

    # Check if restored image size is reasonable
    restored_height = num_slices_height * slice_size
    restored_width = num_slices_width * slice_size
    if restored_height * restored_width > 1e9:  # Warn if image pixels exceed 1 billion
        raise ValueError(
            f"Attempting to allocate excessive image memory: height {restored_height}, width {restored_width}.")

    # Initialize restored image
    restored_img = np.zeros((restored_height, restored_width), dtype=np.uint8)

    # Stitch images
    for i in range(num_slices_height):
        for j in range(num_slices_width):
            slice_filename = f"{i}_{j}.TIF"
            slice_path = os.path.join(input_dir, slice_filename)
            if not os.path.exists(slice_path):
                print(f"Slice file {slice_filename} is missing, skipping.")
                continue

            slice_img = io.imread(slice_path)
            if slice_img.ndim > 2:  # Convert to grayscale if it's an RGB image
                slice_img = cv2.cvtColor(slice_img, cv2.COLOR_RGB2GRAY)

            # Place the slice
            restored_img[i * slice_size:(i + 1) * slice_size,
            j * slice_size:(j + 1) * slice_size] = slice_img

    return restored_img


def stitch_and_crop(crop_dir, input_dir, output_dir):
    """
    Stitch prediction results in each folder and crop padding parts, save as new .tif files.

    Args:
        crop_dir (str): Path to the crop directory containing padding information
        input_dir (str): Input folder path containing subfolders and prediction results.
        output_dir (str): Output folder path for saving stitched .tif files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Traverse each subfolder in the input directory (each subfolder corresponds to one image)
    for foldername in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, foldername, 'DSFA-SwinNet')
        print("folder_path:")
        print(folder_path)
        if not os.path.isdir(folder_path):
            continue

        # Get padding information
        padding_info_path = os.path.join(crop_dir, foldername, 'padding_info.txt')
        print("padding_info_path:")
        print(padding_info_path)
        if not os.path.exists(padding_info_path):
            print(f"Skipping folder {foldername} - No padding info found.")
            continue

        # Read padding_info.txt file
        with open(padding_info_path, 'r') as f:
            padding_info = f.readlines()
            # Extract original size and padded size
            original_size = tuple(map(int, padding_info[0].split(':')[1].strip()[1:-1].split(',')))
            padded_size = tuple(map(int, padding_info[1].split(':')[1].strip()[1:-1].split(',')))

            # Parse padding information
            padding_line = padding_info[2].strip()
            pad_height = int(padding_line.split('Height:')[1].split(',')[0].strip())
            pad_width = int(padding_line.split('Width:')[1].strip())

        stitched_image = restore_image_from_slices(folder_path)

        # Crop out the padding area
        cropped_image = stitched_image[:original_size[0], :original_size[1]]

        # Save the stitched image
        result_path = os.path.join(output_dir, f"{foldername}.TIF")
        # Save image using cv2 to ensure proper display
        cv2.imwrite(result_path, cropped_image)
        print(f"Saved stitched image for {foldername} to {result_path}")

# # Example usage
# input_dir = r"F:\yy\CPVPD\2020PV_shape\train_tiff\pre\predicted"  # Path to prediction results folder
# output_dir = r"F:\yy\CPVPD\2020PV_shape\train_tiff\pre\result"  # Path to output results folder
#
# stitch_and_crop(input_dir, output_dir)
