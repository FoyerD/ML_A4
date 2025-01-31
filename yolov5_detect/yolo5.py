from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt


def eval_yolo():
    # Load a custom trained YOLOv5s model
    model = YOLO("models/yolov5su_trained.pt")
    results = model.val(data="datasets/test.yaml", )
    print("Loss: ", results.fitness)
    print("ACC: ", results.results_dict['metrics/mAP50-95(B)'])
    

def train_yolo():
    # Load a COCO-pretrained YOLOv5n model
    model = YOLO("models/yolov5su.pt")
    results = model.train(data="datasets/dataset.yaml", epochs=10)
    model.save("yolov5su_trained.pt")



def plot_graphs():
    # Extract relevant columns
    # Load the existing CSV file
    csv_file_path = 'runs\\detect\\train\\results.csv'  # Replace with your actual CSV file path
    df = pd.read_csv(csv_file_path)
    epochs = df['epoch']  # Assuming there's an 'epoch' column
    train_classification_loss = df['train/cls_loss']  # Training classification loss
    val_classification_loss = df['val/cls_loss']      # Validation classification loss
    train_classification_accuracy = df['metrics/mAP50-95(B)']  # Training accuracy

    # Create subplots
    plt.figure(figsize=(14, 8))

    # Plot Training and Validation Classification Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_classification_loss, label='Train Loss', color='blue')
    plt.plot(epochs, val_classification_loss, label='Validation Loss', color='orange')
    plt.title('Classification Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # Plot Training and Validation Classification Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_classification_accuracy, label='Train Accuracy', color='green')
    plt.title('Classification Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.savefig("outputs/yolov5_training_plots_final.png")

if __name__ == "__main__":
    plot_graphs()