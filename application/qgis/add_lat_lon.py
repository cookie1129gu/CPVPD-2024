from qgis.PyQt.QtCore import QVariant
from qgis.core import QgsProject, QgsField, QgsGeometry, QgsVectorLayer

# Get all layers in the group2 group
root = QgsProject.instance().layerTreeRoot()
group2 = root.findGroup("sub-group1")

if group2:
    # Iterate through all layers in the group
    for tree_layer in group2.findLayers():
        layer = tree_layer.layer()

        # Check if the layer is valid and is a vector layer
        if not layer.isValid() or not isinstance(layer, QgsVectorLayer):
            print(f"Skipping invalid or non-vector layer: {layer.name()}")
            continue

        print(f"Processing layer: {layer.name()}")

        try:
            # Start editing mode
            if not layer.startEditing():
                print(f"Unable to start editing layer: {layer.name()}")
                continue

            # Add fields if they don't exist
            fields = layer.fields()
            lat_idx = fields.indexFromName('latitude')
            lon_idx = fields.indexFromName('longitude')

            if lat_idx == -1:
                layer.addAttribute(QgsField('latitude', QVariant.Double))
                print(f"Added latitude field to {layer.name()}")
            if lon_idx == -1:
                layer.addAttribute(QgsField('longitude', QVariant.Double))
                print(f"Added longitude field to {layer.name()}")

            # Update field indices
            layer.updateFields()
            lat_idx = layer.fields().indexFromName('latitude')
            lon_idx = layer.fields().indexFromName('longitude')

            # Calculate and populate coordinates
            for feature in layer.getFeatures():
                geom = feature.geometry()
                if geom.isNull() or geom.isEmpty():
                    continue

                # Get geometric center point
                centroid = geom.centroid()
                if not centroid.isEmpty():
                    point = centroid.asPoint()
                    # Update attributes
                    layer.changeAttributeValue(feature.id(), lat_idx, point.y())  # latitude
                    layer.changeAttributeValue(feature.id(), lon_idx, point.x())  # longitude

            # Commit changes
            if layer.commitChanges():
                print(f"Successfully updated coordinate fields for layer {layer.name()}")
            else:
                print(f"Failed to commit changes: {layer.name()}")
                layer.rollBack()

        except Exception as e:
            print(f"Error processing layer {layer.name()}: {str(e)}")
            if layer.isEditable():
                layer.rollBack()
else:
    print("Group1 group not found")

print("Processing completed")
