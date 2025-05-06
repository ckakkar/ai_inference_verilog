# inference.py
# Python baseline for neural network inference

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

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

# Load the trained model
def load_model():
    model = SimpleNN()
    model.load_state_dict(torch.load('python_test/mnist_model.pth'))
    model.eval()
    return model

# Load test images
def load_test_images(filename, num_images=100):
    images = []
    with open(filename, 'rb') as f:
        for _ in range(num_images):
            image = np.frombuffer(f.read(784), dtype=np.uint8)
            images.append(image)
    return images

# Load test labels
def load_test_labels(filename, num_labels=100):
    with open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
    return labels

# Run inference with PyTorch
def run_inference_pytorch(model, images):
    results = []
    predictions = []
    
    start_time = time.time()
    
    for image in images:
        # Convert to float tensor and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = (image_tensor - 0.1307) / 0.3081  # Apply MNIST normalization
        
        # Run inference
        with torch.no_grad():
            output = model(image_tensor)
        
        # Record results and prediction
        results.append(output.numpy())
        predictions.append(np.argmax(output.numpy()))
    
    end_time = time.time()
    
    return results, predictions, end_time - start_time

# Run inference with NumPy (to simulate fixed-point calculation)
def run_inference_numpy(hidden_weights, output_weights, images):
    results = []
    predictions = []
    
    start_time = time.time()
    
    for image in images:
        # Normalize input to match the fixed-point format used in Verilog
        image = image.astype(np.int32)
        
        # First layer
        hidden = np.dot(image, hidden_weights.T)
        hidden_relu = np.maximum(0, hidden)
        
        # Output layer
        output = np.dot(hidden_relu, output_weights.T)
        
        # Record results and prediction
        results.append(output)
        predictions.append(np.argmax(output))
    
    end_time = time.time()
    
    return results, predictions, end_time - start_time

def main():
    # Load model, weights, and test data
    model = load_model()
    
    # Get weights from the model
    hidden_weights = model.fc1.weight.data.numpy()
    output_weights = model.fc2.weight.data.numpy()
    
    # Scale weights to 8-bit range (-128 to 127) to match Verilog simulation
    hidden_weights_scaled = np.clip(hidden_weights * 127, -128, 127).astype(np.int8)
    output_weights_scaled = np.clip(output_weights * 127, -128, 127).astype(np.int8)
    
    # Load test images and labels
    test_images = load_test_images('python_test/mnist_test.bin')
    test_labels = load_test_labels('python_test/mnist_labels.bin')
    
    # Run inference with PyTorch model
    torch_results, torch_predictions, torch_time = run_inference_pytorch(model, test_images)
    
    # Run inference with NumPy (fixed-point simulation)
    numpy_results, numpy_predictions, numpy_time = run_inference_numpy(
        hidden_weights_scaled, output_weights_scaled, test_images)
    
    # Calculate accuracy
    torch_accuracy = np.mean(np.array(torch_predictions) == test_labels) * 100
    numpy_accuracy = np.mean(np.array(numpy_predictions) == test_labels) * 100
    
    print(f"PyTorch inference time: {torch_time:.4f} seconds")
    print(f"PyTorch accuracy: {torch_accuracy:.2f}%")
    print(f"NumPy inference time: {numpy_time:.4f} seconds")
    print(f"NumPy accuracy: {numpy_accuracy:.2f}%")
    
    # Save results for comparison with Verilog
    np.savetxt('python_test/torch_results.txt', np.array(torch_results).reshape(len(torch_results), -1))
    np.savetxt('python_test/numpy_results.txt', np.array(numpy_results).reshape(len(numpy_results), -1))
    
    # Save timing information
    with open('python_test/torch_time.txt', 'w') as f:
        f.write(str(torch_time))
    with open('python_test/numpy_time.txt', 'w') as f:
        f.write(str(numpy_time))
    
    # Print a few sample results
    print("\nSample results (first 5 images):")
    for i in range(5):
        print(f"Image {i}: Label={test_labels[i]}, PyTorch prediction={torch_predictions[i]}, "
              f"NumPy prediction={numpy_predictions[i]}")

if __name__ == "__main__":
    main()