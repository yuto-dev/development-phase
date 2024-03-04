from torchvision import transforms
from dataset.carBrandDataset import CarBrandDataset
from models.cnn import CustomConvNet
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim

# Define your dataset and data loader here
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CarBrandDataset(root_dir=r"dataset\train", transform=transform)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print(f"Number of training samples: {len(train_dataset)}")

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define your model, criterion, and optimizer here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = CustomConvNet(num_classes=7).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop here
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print((i+1) % 100)
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}")
    print("Epoch: ", epoch)

print("Training complete")

# Making Predictions
# Load an image for prediction
from PIL import Image

# Replace 'path_to_your_image.jpg' with the actual path to your image
image_path = r'C:\Users\Abiyyu\Desktop\programming\machineLearning\dela\dataset\train\volkswagen\volkswagen_10.jpg'
image = Image.open(image_path)

# Apply the same transformations as during training
transformed_image = transform(image)

# Convert the image to a PyTorch tensor and add a batch dimension
image_tensor = transformed_image.unsqueeze(0).to(device)

# Make a prediction
model.eval() # Set the model to evaluation mode
with torch.no_grad(): # Temporarily disable gradient tracking
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"Predicted class: {predicted.item()}")

# Evaluating Performance
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the validation images: {100 * correct / total}%')

# Saving the Model
torch.save(model.state_dict(), 'model_weights.pt')
print("Model saved.")


