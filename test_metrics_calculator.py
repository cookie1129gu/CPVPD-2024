import os
import numpy as np
import rasterio
import pandas as pd


def calculate_metrics(label_path, pred_path):
    """
    Calculate precision, recall, F1 score, and intersection over union (IoU)
    """
    with rasterio.open(label_path) as label_ds:
        label = label_ds.read().transpose((1, 2, 0))
    with rasterio.open(pred_path) as pred_ds:
        pred = pred_ds.read().transpose((1, 2, 0))

    label = np.where((label == [255, 255, 255]).all(axis=2), 1, 0)
    pred = np.where((pred == [255, 255, 255]).all(axis=2), 1, 0)

    true_positives = np.sum(np.logical_and(label == 1, pred == 1))
    false_positives = np.sum(np.logical_and(label == 0, pred == 1))
    false_negatives = np.sum(np.logical_and(label == 1, pred == 0))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    intersection = np.sum(np.logical_and(label == 1, pred == 1))
    union = np.sum(np.logical_or(label == 1, pred == 1))
    iou = intersection / union if union > 0 else 0

    return precision, recall, f1, iou


def classify_feature(feat):
    """Return stratified category based on field values"""
    # Elevation stratification
    elev = feat['dem']
    if elev < 500:
        elev_class = 'LEA'
    elif elev < 2000:
        elev_class = 'MEA'
    else:
        elev_class = 'HEA'

    # Land use classification (based on provided mapping)
    lc_map = {
        1: 'Cultivated Land',
        2: 'Woodland',
        3: 'Shrubland',
        4: 'Grassland',
        5: 'Waterbody',
        7: 'Bare Land',
        8: 'Artificial Surface'
    }
    lc_class = lc_map.get(feat['landcover'], 'Unknown')

    return f"{elev_class}_{lc_class}"


def calculate_mean_metrics(df):
    """
    Calculate mean values of Precision, Recall, F1, and IoU for each category
    """
    # Group by category and calculate means
    grouped = df.groupby('class').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'IoU': 'mean'
    }).reset_index()

    # Rename columns
    grouped.columns = ['kind', 'Precision', 'Recall', 'F1', 'IoU']

    # Calculate overall mean and add as last row
    overall_mean = [
        'Overall',
        grouped['Precision'].mean(),
        grouped['Recall'].mean(),
        grouped['F1'].mean(),
        grouped['IoU'].mean()
    ]
    overall_mean_series = pd.Series(overall_mean, index=grouped.columns).to_frame().T
    result_df = pd.concat([grouped, overall_mean_series], ignore_index=True)

    return result_df


def main():
    # Set paths
    label_folder = r".\test_label"
    pred_folder = r".\test_pred_CPVPD-2024"
    excel_path = r".\test_label.xlsx"
    output_folder = r"."

    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Part 1: Calculate metrics for test area labels and datasets
    df = pd.read_excel(excel_path)

    for file in os.listdir(label_folder):
        if file.endswith('.tif'):
            label_path = os.path.join(label_folder, file)
            pred_path = os.path.join(pred_folder, file)

            precision, recall, f1, iou = calculate_metrics(label_path, pred_path)

            # Find corresponding row
            row_index = df[df['test_id'] == int(file.split('.')[0])].index
            if len(row_index) > 0:
                df.at[row_index[0], 'Precision'] = precision
                df.at[row_index[0], 'Recall'] = recall
                df.at[row_index[0], 'F1'] = f1
                df.at[row_index[0], 'IoU'] = iou

    # Part 2: Terrain-land use type classification
    required_columns = ['landcover', 'dem']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in input file: {col}")

    # Classify each row of data
    df['class'] = df.apply(classify_feature, axis=1)

    # Part 3: Calculate mean metrics
    result_df = calculate_mean_metrics(df)

    # Save result
    output_file = os.path.join(output_folder, 'test_metrics.xlsx')
    result_df.to_excel(output_file, index=False)

    print(f"Calculation completed and results saved to {output_file}")


if __name__ == "__main__":
    main()
