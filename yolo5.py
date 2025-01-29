from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt


def eval_yolo():
    # Load a custom trained YOLOv5s model
    model = YOLO("models/yolov5su_trained.pt")
    results = model.val(data="datasets/test.yaml", )
    precision = results.results_dict['metrics/precision(B)']  # Adjust column names as necessary
    recall = results.results_dict['metrics/recall(B)']        # Adjust column names as necessary
    N = 128

    # Calculate True Positives, False Positives, and False Negatives
    TP = precision * N
    FP = (1 - precision) * N
    FN = (1 - recall) * (TP / recall) if recall > 0 else 0  # Avoid division by zero

    # Calculate accuracy
    accuracy = (N-FP-FN) / N
    
    print("Loss: ", results.fitness)
    print("ACC: ", accuracy)
    

def train_yolo():
    # Load a COCO-pretrained YOLOv5n model
    model = YOLO("models/yolov5su.pt")
    results = model.train(data="datasets/dataset.yaml", epochs=10)
    model.save("yolov5su_trained.pt")

def add_acc():

    # Load the existing CSV file
    csv_file_path = 'runs\\detect\\train\\results'  # Replace with your actual CSV file path
    df = pd.read_csv(csv_file_path + '.csv')

    # Initialize a list to store classification accuracy
    accuracy_list = []
    val_accuracy_list = []

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        precision = row['metrics/precision(B)']  # Adjust column names as necessary
        recall = row['metrics/recall(B)']        # Adjust column names as necessary
        val_precision = row['metrics/mAP50(B)']  # Adjust column names as necessary
        val_recall = row['metrics/mAP50-95(B)']        # Adjust column names as necessary
        N = 16 * 5

        # Calculate True Positives, False Positives, and False Negatives
        TP = precision * N
        FP = (1 - precision) * N
        FN = (1 - recall) * (TP / recall) if recall > 0 else 0  # Avoid division by zero

        # Calculate accuracy
        accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        accuracy_list.append(accuracy)
        
        # Calculate True Positives, False Positives, and False Negatives
        val_TP = val_precision * N
        val_FP = (1 - val_precision) * N
        val_FN = (1 - val_recall) * (val_TP / val_recall) if val_recall > 0 else 0  # Avoid division by zero

        
        val_accuracy = val_TP / (val_TP + val_FP + val_FN) if (val_TP + val_FP + val_FN) > 0 else 0
        val_accuracy_list.append(val_accuracy)

    # Add the new accuracy column to the DataFrame
    df['classification_accuracy'] = accuracy_list
    df['val_accuracy'] = val_accuracy_list

    # Save the updated DataFrame back to CSV
    df.to_csv(csv_file_path + 'new.csv', index=False)

def plot_graphs():
    # Extract relevant columns
    # Load the existing CSV file
    csv_file_path = 'runs\\detect\\train\\resultsnew.csv'  # Replace with your actual CSV file path
    df = pd.read_csv(csv_file_path)
    epochs = df['epoch']  # Assuming there's an 'epoch' column
    train_classification_loss = df['train/cls_loss']  # Training classification loss
    val_classification_loss = df['val/cls_loss']      # Validation classification loss
    train_classification_accuracy = df['classification_accuracy']  # Training accuracy
    val_classification_accuracy = df['val_accuracy']  # Validation accuracy (you may need to adjust this)

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
    plt.plot(epochs, val_classification_accuracy, label='Validation Accuracy', color='red')
    plt.title('Classification Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.savefig("outputs/yolov5_training_plots_final.png")

if __name__ == "__main__":
    eval_yolo()