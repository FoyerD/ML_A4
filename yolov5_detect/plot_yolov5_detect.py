import pandas as pd
import matplotlib.pyplot as plt

def plot_graphs(parth_to_csv):
    # Load the CSV data
    data = pd.read_csv(parth_to_csv)
    data.columns = data.columns.str.strip()
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plotting Accuracy
    ax1.plot(data['epoch'], data['train/acc'], label='Train Accuracy', color='blue')
    ax1.plot(data['epoch'], data['valid/acc'], label='Valid Accuracy', color='orange')
    ax1.set_title('Accuracy vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid()

    # Plotting Loss
    ax2.plot(data['epoch'], data['train/loss'], label='Train Loss', color='red')
    ax2.plot(data['epoch'], data['valid/loss'], label='Valid Loss', color='green')
    ax2.set_title('Loss vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid()

    # Show the plots
    plt.tight_layout()
    plt.savefig('outputs/yolov5_cls_training_plots.png')


if __name__ == "__main__":
    plot_graphs('runs/train-cls/exp/results.csv')  