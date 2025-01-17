import os
import yaml
import numpy as np
import pandas as pd
import cv2
from PIL import Image

def compute_ground_truth_mask(watermarked_image, clean_image, threshold=0.1):
    diff = np.abs(np.array(watermarked_image) - np.array(clean_image)) / 255.0
    mask = (diff > threshold).astype(np.float32)
    mask = np.max(mask, axis=2)  # Combine channels
    return mask

def create_annotations(dirs_list, output_dir, threshold=0.1):
    """
    dirs_list: [dir1, dir2, ...]
        Each directory contains a metadata.csv file.
    All metadata is combined, shuffled, and saved under a single output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    # Combine all metadata
    combined_data = []

    for d in dirs_list:
        metadata_path = os.path.join(d, "metadata.csv")
        df = pd.read_csv(metadata_path)
        df['root_dir'] = d
        combined_data.append(df)

    combined_data = pd.concat(combined_data, ignore_index=True)

    # Shuffle the data
    combined_data = combined_data.sample(frac=1).reset_index(drop=True)

    for idx, row in combined_data.iterrows():
        root_dir = row['root_dir']
        clean_path = os.path.join(root_dir, row.iloc[0])
        watermarked_path = os.path.join(root_dir, row.iloc[1])

        clean_image = Image.open(clean_path).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
        watermarked_image = Image.open(watermarked_path).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)

        mask = compute_ground_truth_mask(watermarked_image, clean_image, threshold=threshold)
        mask_np = mask

        contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotations = []
        for contour in contours:
            if cv2.contourArea(contour) < 10:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            class_id = 0
            x_center = (x + w / 2) / mask_np.shape[1]
            y_center = (y + h / 2) / mask_np.shape[0]
            width = w / mask_np.shape[1]
            height = h / mask_np.shape[0]
            annotations.append([class_id, x_center, y_center, width, height])

        label_path = os.path.join(output_dir, 'labels', f"{idx}.txt")
        if len(annotations) == 0:
            with open(label_path, 'w') as f:
                pass
        else:
            with open(label_path, 'w') as f:
                for anno in annotations:
                    f.write(' '.join([str(a) for a in anno]) + '\n')

        image_filename = f"{idx}.jpg"
        image_save_path = os.path.join(output_dir, 'images', image_filename)
        watermarked_image.save(image_save_path)

    print(f"Annotations created and saved to {output_dir}.")

def create_yaml_file(train_annotations_dir, val_annotations_dir, yaml_path):
    data_yaml = {
        'train': os.path.abspath(os.path.join(train_annotations_dir, 'images')),
        'val': os.path.abspath(os.path.join(val_annotations_dir, 'images')),
        'nc': 1,
        'names': ['watermark']
    }
    with open(yaml_path, 'w') as file:
        yaml.dump(data_yaml, file)
    print(f"YAML file created: {yaml_path}")

if __name__ == "__main__":
    train_dir = ["no_logo_and_low_opacity_watermark_dataset_train", 
                 "no_logo_and_high_opacity_watermark_dataset_train", 
                 "logo_and_high_opacity_watermark_dataset_train"]
    validation_dir = ["no_logo_and_low_opacity_watermark_dataset_test", 
                      "no_logo_and_high_opacity_watermark_dataset_test", 
                      "logo_and_high_opacity_watermark_dataset_test"]
    annotations_root = "yolo_watermarked_dataset"
    yaml_filename = "watermark_dataset.yaml"

    train_annotations_dir = f"{annotations_root}/{annotations_root}_train"
    print("Creating annotations for training data...")
    create_annotations(
        dirs_list=train_dir,
        output_dir=train_annotations_dir
    )

    val_annotations_dir = f"{annotations_root}/{annotations_root}_val"
    print("Creating annotations for validation data...")
    create_annotations(
        dirs_list=validation_dir,
        output_dir=val_annotations_dir
    )

    yaml_path = os.path.join(annotations_root, yaml_filename)
    print("Creating YAML file...")
    create_yaml_file(train_annotations_dir, val_annotations_dir, yaml_path)
