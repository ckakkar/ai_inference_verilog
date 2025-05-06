# AI Inference Accelerator in Verilog

This project implements a simple neural network inference accelerator in Verilog, simulated using Verilator. The design includes a fully connected neural network with one hidden layer and ReLU activation function, trained on the MNIST dataset for digit classification.

## Project Structure

- `src/`: Verilog source files
  - `mac_unit.v`: Multiply-accumulate unit 
  - `dense_layer.v`: Dense (fully connected) layer
  - `relu_activation.v`: ReLU activation function
  - `top.v`: Top-level module connecting all components
- `sim/`: Simulation files
  - `main.cpp`: C++ testbench for Verilator
- `python_test/`: Python baseline implementation
  - `train_model.py`: Trains a simple model and exports weights
  - `inference.py`: Python baseline implementation
  - `compare_results.py`: Compares Verilog simulation results with Python
- `weights/`: Contains exported model weights
- `Makefile`: Build automation

## Requirements

The following tools are required to run this project:

- **Verilator**: HDL simulator
- **C++ Compiler**: For Verilator testbench
- **Python 3**: For training and baseline comparison
- **PyTorch**: For neural network implementation
- **NumPy**: For numerical computing
- **Matplotlib**: For plotting results

## Setup Instructions for Mac M2

### 1. Install Dependencies

Using Homebrew:

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Verilator
brew install verilator

# Install Python dependencies
pip3 install torch torchvision numpy matplotlib
```

### 2. Setup Project Structure

Create the project structure and add all the files as shown in this conversation.

```bash
# Create the main project directory
mkdir -p ai_inference_verilog
cd ai_inference_verilog

# Create subdirectories
mkdir -p src sim python_test weights
```

Add all the source files to their respective directories.

## Running the Project

### 1. Build and Run Everything

```bash
make all
```

This will:
1. Verilate the design
2. Train the neural network and export weights
3. Run the Verilog simulation
4. Compare results with the Python baseline

### 2. Run Individual Steps

Train the model and export weights:
```bash
make train
```

Run the Python baseline inference:
```bash
make python_inference
```

Run the Verilog simulation:
```bash
make simulate
```

Compare results:
```bash
make compare
```

### 3. Clean Build Artifacts

```bash
make clean
```

## Understanding the Results

After running the project, several output files will be generated:

- `verilog_results.txt`: Results from the Verilog simulation
- `python_test/torch_results.txt`: Results from PyTorch inference
- `python_test/numpy_results.txt`: Results from NumPy inference
- `comparison_results.png`: Visualization comparing outputs
- `performance_comparison.png`: Comparison of inference time and accuracy
- `trace.vcd`: Waveform trace file (can be viewed with GTKWave)

## Extension Ideas

1. **Optimize for performance**:
   - Add pipeline stages to increase throughput
   - Implement parallelism with multiple MAC units

2. **Improve accuracy**:
   - Use higher precision fixed-point representation
   - Implement more complex activation functions

3. **Add more features**:
   - Implement a convolutional layer
   - Add batch normalization
   - Implement a more complex network architecture