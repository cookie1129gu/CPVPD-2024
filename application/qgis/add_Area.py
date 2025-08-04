from qgis.PyQt.QtCore import QVariant
from qgis.core import (
    QgsProject,
    QgsField,
    QgsVectorLayer,
    QgsWkbTypes,
    QgsExpression,
    QgsExpressionContext,
    QgsExpressionContextUtils,
    QgsFeatureRequest
)

# Get the root node of the layer tree
root = QgsProject.instance().layerTreeRoot()

# Find the layer group named "sub-group1"
target_group = root.findGroup("sub-group1")

if target_group:
    print(f"Starting processing layers in 'group1' group...")

    # Iterate through all layers in the group
    for layer in target_group.findLayers():
        # Get the actual vector layer object
        vlayer = layer.layer()

        # Check if it's a vector layer
        if not isinstance(vlayer, QgsVectorLayer):
            print(f"Skipping non-vector layer: {vlayer.name()}")
            continue

        # Check if the layer is of polygon type
        if vlayer.geometryType() != QgsWkbTypes.PolygonGeometry:
            print(f"Skipping non-polygon layer: {vlayer.name()}")
            continue

        # Verify if the coordinate system is ESRI:102025
        if vlayer.crs().authid() != "ESRI:102025":
            print(f"Warning: The coordinate system of layer {vlayer.name()} is not ESRI:102025, currently {vlayer.crs().authid()}")
            # Continue processing but it may affect area calculation accuracy

        # Start editing session
        vlayer.startEditing()
        print(f"Processing layer: {vlayer.name()}")

        # Add Area field if it doesn't exist
        field_name = "Area"
        field_index = vlayer.fields().indexFromName(field_name)

        if field_index == -1:  # Field doesn't exist
            new_field = QgsField(field_name, QVariant.Double, len=20, prec=3)
            vlayer.addAttribute(new_field)
            vlayer.updateFields()  # Update fields
            field_index = vlayer.fields().indexFromName(field_name)
            print(f"Added field '{field_name}'")

        # Create expression to calculate $area
        expression = QgsExpression('$area')
        context = QgsExpressionContext()
        context.appendScopes(QgsExpressionContextUtils.globalProjectLayerScopes(vlayer))

        # Calculate and populate area
        features = vlayer.getFeatures()
        vlayer.beginEditCommand("Calculate area")  # Supports undo operation

        for feature in features:
            context.setFeature(feature)
            # Calculate $area value
            area = expression.evaluate(context)

            # Update attribute
            vlayer.changeAttributeValue(feature.id(), field_index, area)

        vlayer.endEditCommand()  # End edit command
        vlayer.commitChanges()  # Commit changes
        print(f"Layer {vlayer.name()} processing completed, calculated area for {vlayer.featureCount()} features\n")

else:
    print("Layer group named 'group1' not found")

print("All layers processed!")
