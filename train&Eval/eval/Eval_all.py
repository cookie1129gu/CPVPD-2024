import os
import shutil
from Eval.pre.train.datasets.Dataset_load_datu import RS
from EvalSeg_datu import evalSeg_datu

# Root directory
root = r'D:/desk/PV/ImageProcess/Eval/result/test/'

processed_path = os.path.join(root, 'processed')
output_base_path = os.path.join(root, 'predicted')
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)


def main():
    # Iterate through each subfolder
    for folder in os.listdir(processed_path):
        folder_path = os.path.join(processed_path, folder)

        if not os.path.isdir(folder_path):  # Skip non-folder items
            continue

        # Update root for RS
        root = folder_path
        print("root:")  # test
        print(root)  # test

        # Initialize RS data object
        dataset = RS("test", root)

        pred_path = os.path.join(output_base_path, folder, 'DSFA-SwinNet')
        os.makedirs(pred_path, exist_ok=True)
        print("pred_path:")  # test
        print(pred_path)  # test

        # Call evalSeg_datu for prediction
        evalSeg_datu(dataset, pred_path)  # evalSeg_datu is your prediction function

        # Copy txt files from subfolder to prediction folder
        txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
        for txt in txt_files:
            shutil.copy(os.path.join(folder_path, txt), pred_path)

        print(f"Processed folder: {folder}")


if __name__ == '__main__':
    # Ensure execution only when running as main program
    main()
