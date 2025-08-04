import os
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProcessingFeedback, QgsProject, edit
import processing
from processing.core.Processing import Processing

# Initialize Processing
Processing.initialize()

# Set input and output paths
root = 'D:/desk/PV/test/result/'

input_path = os.path.join(root, 'clear_tif')

output_path = os.path.join(root, 'shp_result')
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Get all TIF files
tif_files = [f for f in os.listdir(input_path) if f.endswith('.TIF')]

# Processing feedback
feedback = QgsProcessingFeedback()

# Iterate through all TIF files and convert
for tif in tif_files:
    tif_path = os.path.join(input_path, tif)
    shp_name = os.path.splitext(tif)[0] + ".shp"
    shp_path = os.path.join(output_path, shp_name)

    # Use GDAL to convert raster to vector, keeping all pixels
    params = {
        'INPUT': tif_path,
        'BAND': 1,  # Process the first band
        'FIELD': 'DN',
        'EIGHT_CONNECTEDNESS': False,
        'OUTPUT': shp_path
    }

    processing.run("gdal:polygonize", params, feedback=feedback)

    # Load the vector file
    vector_layer = QgsVectorLayer(shp_path, "temp_layer", "ogr")

    # Enable editing mode to delete features
    with edit(vector_layer):
        # Iterate through all features and delete those with DN not equal to 255 (non-white pixels)
        for feature in vector_layer.getFeatures():
            if feature["DN"] != 255:  # Only keep white pixels with DN = 255
                vector_layer.deleteFeature(feature.id())

    # Save changes
    vector_layer.commitChanges()

    print(f"Conversion completed, retaining white pixels: {shp_path}")

print("All TIF files have been successfully converted to SHP files, retaining only white pixels!")
