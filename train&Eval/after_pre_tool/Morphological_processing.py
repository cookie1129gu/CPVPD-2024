import os
import cv2
import numpy as np
from tqdm import tqdm


def filter_rectangular_objects(image):
    MIN_RECT_RATIO = 2  # Minimum aspect ratio for elongated rectangles
    MIN_AREA = 50  # Filter out areas smaller than this value (50, 500)
    ANGLE_TOLERANCE = 10  # Allowed angle deviation (from horizontal/vertical)

    # Convert to binary image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Filtering rules
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]

        # Filter out too small areas
        if area < MIN_AREA:
            continue

        # Calculate aspect ratio
        aspect_ratio = max(w, h) / max(1, min(w, h))
        if aspect_ratio < MIN_RECT_RATIO:
            continue

        # Calculate angle of minimum enclosing rectangle
        rect = cv2.minAreaRect(np.column_stack(np.where(labels == i)))
        angle = abs(rect[-1])  # Get angle
        if angle > 90:
            angle = abs(angle - 180)

        # Angle filtering
        if min(abs(angle), abs(angle - 90)) > ANGLE_TOLERANCE:
            continue

        # Passed filtering, keep this area
        mask[labels == i] = 255

    return mask


def process_images(input_path, output_path):
    # Ensure output path exists
    if not os.path.exists(output_path): os.makedirs(output_path)

    for filename in tqdm(os.listdir(input_path)):
        # print(filename)
        if filename.endswith('.TIF'):
            image_path = os.path.join(input_path, filename)
            output_image_path = os.path.join(output_path, filename)

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            filtered_image = filter_rectangular_objects(image)

            cv2.imwrite(output_image_path, filtered_image)

    print("Processing complete.")


def process_subfolders(input_dir, output_dir):
    # Traverse each subfolder in the input directory
    for foldername in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, foldername, 'DSFA-SwinNet')

        # Check if the subfolder exists
        if os.path.isdir(folder_path):
            # Construct output path
            output_path = os.path.join(output_dir, foldername, 'DSFA-SwinNet')
            process_images(folder_path, output_path)  # Call process_images for each subfolder
        else:
            print(f'{folder_path} does not exist')

# if __name__ == "__main__":
#     # Constant definitions
#     INPUT_PATH = r'D:\desk\PV\tiff\beijing\beijing_all\result_no_geoinfo'
#     OUTPUT_PATH = r'D:\desk\PV\tiff\beijing\beijing_all\processed_result'
#     process_images(INPUT_PATH, OUTPUT_PATH)
#     print("Processing complete.")
