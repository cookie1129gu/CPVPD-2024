import os
import cv2
import numpy as np
from osgeo import gdal



def gamma_correction(image, gamma=1.0):
    """Apply gamma correction to the image"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_image(input_path, output_path, gamma=1.2):
    """Apply histogram equalization and gamma correction to the image"""
    # Open the image file
    dataset = gdal.Open(input_path)
    if dataset is None:
        print(f"Unable to open image file: {input_path}")
        return

    # Read channel data, assuming the image is 4-channel (RGBA), we only process RGB channels
    r = dataset.GetRasterBand(1).ReadAsArray()
    g = dataset.GetRasterBand(2).ReadAsArray()
    b = dataset.GetRasterBand(3).ReadAsArray()

    # Apply histogram equalization to each channel
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # Merge channels
    equalized_image = cv2.merge((r_eq, g_eq, b_eq))

    # Apply gamma correction
    gamma_corrected_image = gamma_correction(equalized_image, gamma=gamma)

    # Create output image, preserving geographic information
    driver = gdal.GetDriverByName('GTiff')
    out_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 3, gdal.GDT_Byte)
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    # Write processed RGB channels
    out_dataset.GetRasterBand(1).WriteArray(gamma_corrected_image[:, :, 0])
    out_dataset.GetRasterBand(2).WriteArray(gamma_corrected_image[:, :, 1])
    out_dataset.GetRasterBand(3).WriteArray(gamma_corrected_image[:, :, 2])

    # Release resources
    out_dataset.FlushCache()
    del out_dataset
    del dataset
    print(f"Processed image file: {input_path}")

def main(tiff_path, output_folder):
    # Iterate through image files in the tiff folder
    for filename in os.listdir(tiff_path):
        if filename.endswith('.result') and not filename.startswith('label'):
            input_filepath = os.path.join(tiff_path, filename)
            output_filepath = os.path.join(output_folder, filename)

            # Process image and save
            process_image(input_filepath, output_filepath)

    print("All images have been processed with gamma correction and histogram equalization!")
