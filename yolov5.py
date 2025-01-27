from ultralytics import YOLO


def eval_yolo():
    # Load a custom trained YOLOv5s model
    model = YOLO("models/yolov8n-cls_trained.pt")

    # Display model information (optional)
    model.info()

    # Evaluate the model on the COCO8 validation dataset
    results = model.val(data="datasets")

    # Print results
    print(results)

def train_yolo():
    # Load a COCO-pretrained YOLOv5n model
    model = YOLO("models/yolov8n-cls.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data="datasets", epochs=5)

    model.save("models/yolov8n-cls_trained.pt")

if __name__ == "__main__":
    eval_yolo()