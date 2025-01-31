import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('vgg_training_metrics.csv')

# Extract data from the DataFrame
epochs = df['Epoch']
train_loss = df['Train Loss']
val_loss = df['Val Loss']
train_accuracy = df['Train Accuracy (%)']
val_accuracy = df['Val Accuracy (%)']


# Create a figure with two subplots (1 row, 2 columns)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot Loss
ax1.set_title('Loss per Epoch')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red', marker='o')
ax1.plot(epochs, val_loss, label='Val Loss', color='tab:orange', marker='o')
ax1.legend()
ax1.grid()

# Plot Accuracy
ax2.set_title('Accuracy per Epoch')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.plot(epochs, train_accuracy, label='Train Accuracy (%)', color='tab:blue', marker='x')
ax2.plot(epochs, val_accuracy, label='Val Accuracy (%)', color='tab:cyan', marker='x')
ax2.legend()
ax2.grid()

# Show the plots
plt.tight_layout()
plt.savefig("outputs/yolov5_training_plots_final.png")
plt.show()
