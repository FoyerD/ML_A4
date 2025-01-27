import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")
        
        # Read the corresponding label from the text file
        label_name = os.path.splitext(self.image_filenames[idx])[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())

        if self.transform:
            image = self.transform(image)

        return image, label-1

# Hyperparameters
batch_size = 32
num_classes = 102 
num_epochs = 15
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets
train_dataset = CustomImageDataset(
    image_dir='data/images/train',
    label_dir='data/labels/train',
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = CustomImageDataset(
    image_dir='data/images/val',
    label_dir='data/labels/val',
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load pre-trained VGG19 model
model = models.vgg19(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier
model.classifier[6] = nn.Linear(4096, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

train_loss_per_epoch = []
train_accuracy_per_epoch = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_loss_per_epoch.append(epoch_loss)
    train_accuracy_per_epoch.append(epoch_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Save loss and accuracy to CSV
results_df = pd.DataFrame({'Epoch': range(1, num_epochs + 1), 
                            'Loss': train_loss_per_epoch, 
                            'Accuracy': train_accuracy_per_epoch})
results_df.to_csv("training_results.csv", index=False)

# Validation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the validation set: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'vgg19_transfer_learning.pth')

# Plotting the training loss and accuracy per epoch
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_per_epoch, marker='o', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss per Epoch')
plt.xticks(range(1, num_epochs + 1))
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracy_per_epoch, marker='o', color='orange', label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy per Epoch')
plt.xticks(range(1, num_epochs + 1))
plt.legend()

plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()
