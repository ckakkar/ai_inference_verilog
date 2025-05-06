# train_model.py
# Train a simple neural network on MNIST and export weights

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Directory for weights
os.makedirs('weights', exist_ok=True)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=16, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize model
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/100:.3f}')
                running_loss = 0.0

# Test the model
def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    return correct / total

# Train and test
print("Training the model...")
train(model, train_loader, criterion, optimizer)
accuracy = test(model, test_loader)

# Export weights
def export_weights(model):
    # Convert weights to 8-bit fixed point representation (Q7.0)
    # Scale to range [-128, 127]
    
    # First layer (784 x 16)
    fc1_weight = model.fc1.weight.data.numpy()
    fc1_weight_scaled = np.clip(fc1_weight * 127, -128, 127).astype(np.int8)
    
    # Second layer (16 x 10)
    fc2_weight = model.fc2.weight.data.numpy()
    fc2_weight_scaled = np.clip(fc2_weight * 127, -128, 127).astype(np.int8)
    
    # Save as text files
    with open('weights/hidden_weights.txt', 'w') as f:
        for neuron in fc1_weight_scaled:
            for weight in neuron:
                # Write as 2-digit hex
                f.write(f'{weight & 0xff:02x} ')
            f.write('\n')
    
    with open('weights/output_weights.txt', 'w') as f:
        for neuron in fc2_weight_scaled:
            for weight in neuron:
                f.write(f'{weight & 0xff:02x} ')
            f.write('\n')
    
    print("Weights exported to weights/hidden_weights.txt and weights/output_weights.txt")
    
    # Save some test images for verification
    test_images = []
    test_labels = []
    
    for i, (inputs, labels) in enumerate(test_loader):
        if i == 0:  # Just take the first batch
            # Convert to 8-bit unsigned format (0-255)
            images = inputs.view(-1, 28, 28).numpy() * 255
            images = np.clip(images, 0, 255).astype(np.uint8)
            
            # Save first 100 images
            num_save = min(100, len(images))
            os.makedirs('python_test', exist_ok=True)
            with open('python_test/mnist_test.bin', 'wb') as f:
                for idx in range(num_save):
                    image = images[idx].flatten()
                    f.write(bytes(image))
                    test_images.append(image)
                    test_labels.append(labels[idx].item())
            
            # Save corresponding labels
            with open('python_test/mnist_labels.bin', 'wb') as f:
                f.write(bytes([labels[idx].item() for idx in range(num_save)]))
            
            print(f"Saved {num_save} test images and labels to mnist_test.bin and mnist_labels.bin")
            break
    
    return test_images, test_labels

# Export the weights and test data
test_images, test_labels = export_weights(model)

# Save the model
torch.save(model.state_dict(), 'python_test/mnist_model.pth')
print("Model saved to python_test/mnist_model.pth")

# Plot a few test images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {test_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.savefig('python_test/test_images.png')
plt.close()

print("Sample test images saved to python_test/test_images.png")