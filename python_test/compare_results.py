# compare_results.py
# Compare results from Verilog simulation with Python baseline

import numpy as np
import matplotlib.pyplot as plt

# Load results
def load_results(filename):
    return np.loadtxt(filename)

# Calculate accuracy
def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels) * 100

# Load test labels
def load_test_labels(filename, num_labels=100):
    with open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(num_labels), dtype=np.uint8)
    return labels

def main():
    # Load results
    verilog_results = load_results('verilog_results.txt')
    numpy_results = load_results('python_test/numpy_results.txt')
    torch_results = load_results('python_test/torch_results.txt')
    
    # Load test labels
    test_labels = load_test_labels('python_test/mnist_labels.bin')
    
    # Convert results to predictions
    verilog_predictions = np.argmax(verilog_results, axis=1)
    numpy_predictions = np.argmax(numpy_results, axis=1)
    torch_predictions = np.argmax(torch_results, axis=1)
    
    # Calculate accuracy
    verilog_accuracy = calculate_accuracy(verilog_predictions, test_labels)
    numpy_accuracy = calculate_accuracy(numpy_predictions, test_labels)
    torch_accuracy = calculate_accuracy(torch_predictions, test_labels)
    
    print(f"Verilog accuracy: {verilog_accuracy:.2f}%")
    print(f"NumPy accuracy: {numpy_accuracy:.2f}%")
    print(f"PyTorch accuracy: {torch_accuracy:.2f}%")
    
    # Compare differences
    verilog_vs_numpy = np.mean(verilog_predictions == numpy_predictions) * 100
    verilog_vs_torch = np.mean(verilog_predictions == torch_predictions) * 100
    
    print(f"Verilog vs NumPy agreement: {verilog_vs_numpy:.2f}%")
    print(f"Verilog vs PyTorch agreement: {verilog_vs_torch:.2f}%")
    
    # Plot comparison of output values
    plt.figure(figsize=(12, 6))
    
    # Randomly select a few samples to visualize
    sample_indices = np.random.choice(len(verilog_results), 5, replace=False)
    
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, 3, i+1)
        
        x = np.arange(10)
        width = 0.25
        
        plt.bar(x - width, verilog_results[idx], width, label='Verilog')
        plt.bar(x, numpy_results[idx], width, label='NumPy')
        plt.bar(x + width, torch_results[idx], width, label='PyTorch')
        
        plt.xlabel('Digit')
        plt.ylabel('Output value')
        plt.title(f'Sample {idx} (Label: {test_labels[idx]})')
        plt.xticks(x)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    plt.close()
    
    print("Comparison plot saved to comparison_results.png")
    
    # Plot performance comparison
    try:
        with open('verilog_time.txt', 'r') as f:
            verilog_time = float(f.read().strip())
        with open('python_test/numpy_time.txt', 'r') as f:
            numpy_time = float(f.read().strip())
        with open('python_test/torch_time.txt', 'r') as f:
            torch_time = float(f.read().strip())
            
        plt.figure(figsize=(10, 5))
        
        # Timing comparison
        plt.subplot(1, 2, 1)
        plt.bar(['Verilog', 'NumPy', 'PyTorch'], [verilog_time, numpy_time, torch_time])
        plt.ylabel('Inference time (seconds)')
        plt.title('Inference Time Comparison')
        
        # Accuracy comparison
        plt.subplot(1, 2, 2)
        plt.bar(['Verilog', 'NumPy', 'PyTorch'], [verilog_accuracy, numpy_accuracy, torch_accuracy])
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy Comparison')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        plt.close()
        
        print("Performance comparison plot saved to performance_comparison.png")
    except:
        print("Timing information not available. Skipping performance comparison plot.")

if __name__ == "__main__":
    main()