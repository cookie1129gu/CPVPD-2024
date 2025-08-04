import os
from osgeo import gdal


def copy_geoinfo(source_path, target_path):
    """
    Copy geospatial information from source_path to target_path.
    """
    # Open source file (with geospatial information)
    source_ds = gdal.Open(source_path)
    if source_ds is None:
        print(f"Unable to open source file: {source_path}")
        return

    # Open target file (to be updated with geospatial information)
    target_ds = gdal.Open(target_path, gdal.GA_Update)
    if target_ds is None:
        print(f"Unable to open target file: {target_path}")
        return

    # Get geospatial information from source file
    geo_transform = source_ds.GetGeoTransform()  # Affine transformation parameters
    projection = source_ds.GetProjection()  # Projection information

    # Write geospatial information to target file
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)

    # Close files
    source_ds = None
    target_ds = None

    print(f"Successfully copied geospatial information from {source_path} to {target_path}")


def copy_geoinfo_from_folder(source_folder, target_folder):
    """
    Copy geospatial information from all result files in source_folder to corresponding files in target_folder.
    """
    # Get all result files in source_folder
    source_files = [f for f in os.listdir(source_folder) if f.endswith('.tif')]

    for source_file in source_files:
        source_file_path = os.path.join(source_folder, source_file)

        # Find corresponding file name in target_folder (assuming file names are the same)
        target_file_path = os.path.join(target_folder, source_file)

        if os.path.exists(target_file_path):
            # Copy geospatial information
            copy_geoinfo(source_file_path, target_file_path)
        else:
            print(f"Target file does not exist: {target_file_path}")

# # Folder paths
# source_folder = r"F:\yy\CPVPD\2020PV_shape\train_tiff\pre\result"  # Source folder
# target_folder = r"F:\yy\CPVPD\2020PV_shape\train_tiff\pre\result"  # Target folder
#
# # Call function to copy geospatial information for all files
# copy_geoinfo_from_folder(source_folder, target_folder)
#
# print("finish")
