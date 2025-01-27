import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR

from WSSS.datamodules.fgbg_datamodule import ForegroundTextureDataModule
from WSSS.core.resnet import resnet50

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224))])

# Load your dataset (replace 'path_to_dataset' with the actual path)
module = ForegroundTextureDataModule(transforms=transform)
train_loader, val_loader, test_loader = module.return_dataloaders()

# Load pre-trained ResNet-50 model
model = resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Modify the final fully connected layer for two classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 20)  # Assuming 20 classes

# Send the model to the device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
# Training loop
num_epochs = 20
min_val_loss = torch.inf
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for (fg_images, bg_images, masks,
         bg_bg_labels, bg_fg_labels,
         labels) in train_loader:
        inputs = torch.cat([fg_images, bg_images])
        # labels = torch.cat(
        #     [torch.ones(fg_images.shape[0], dtype=torch.uint8), torch.zeros(bg_images.shape[0], dtype=torch.uint8)])
        labels = torch.cat([labels, bg_bg_labels + 10])
        inputs, labels = inputs.to(device).float(), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[1], labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    correct = 0
    total = 0
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for (fg_images, bg_images, masks,
             bg_bg_labels, bg_fg_labels,
             labels) in val_loader:
            inputs = torch.cat([fg_images, bg_images])
            # labels = torch.cat(
            #    [torch.ones(fg_images.shape[0], dtype=torch.uint8), torch.zeros(bg_images.shape[0], dtype=torch.uint8)])
            labels = torch.cat([labels, bg_bg_labels + 10])
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)[1]
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        val_loss = val_loss / total
        print(f"Validation Accuracy: {accuracy}")
        if val_loss < min_val_loss:
            print(f"Validation loss has decreased in {epoch}, from {min_val_loss} --> {val_loss}. Saving the model...")
            min_val_loss = val_loss
            torch.save(model.state_dict(), "texture_fashionmnist")

    scheduler.step(val_loss)
    model.train()

# Evaluation on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for (fg_images, bg_images, masks,
         bg_bg_labels, bg_fg_labels,
         labels) in test_loader:
        inputs = torch.cat([fg_images, bg_images])
        # labels = torch.cat(
        #    [torch.ones(fg_images.shape[0], dtype=torch.uint8), torch.zeros(bg_images.shape[0], dtype=torch.uint8)])
        labels = torch.cat([labels, bg_fg_labels + 10])
        inputs, labels = inputs.to(device).float(), labels.to(device)
        outputs = model(inputs)[1]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

torch.save(model.state_dict(), "texture")
accuracy = correct / total
print(f"Test Accuracy: {accuracy}")
