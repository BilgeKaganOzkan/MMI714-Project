import os
from ultralytics import YOLO

def train_yolo_model(dataset_yaml, dataset_dir, save_dir, epochs=10, imgsz=128):
    model = YOLO("yolov8s.pt")
    save_path = os.path.join(save_dir, f"yolo_{os.path.basename(dataset_dir)}_best.pt")
    model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz)
    model.save(save_path)
    print(f"Best model saved: {save_path}")
    return save_path

if __name__ == "__main__":
    annotations_root = "yolo_watermarked_dataset"
    model_save_dir = "yolo_best_model"
    yaml_filename = "watermark_dataset.yaml"
    train_annotations_dir = f"{annotations_root}/{annotations_root}_train"
    val_annotations_dir = f"{annotations_root}/{annotations_root}_val"
    yaml_path = os.path.join(annotations_root, yaml_filename)

    print("Training YOLO model...")
    os.makedirs(model_save_dir, exist_ok=True)
    train_yolo_model(yaml_path, train_annotations_dir, save_dir=model_save_dir, epochs=50, imgsz=128)
    print("Training completed successfully.")