# Makefile for AI Inference Accelerator in Verilog

# Directories
SRC_DIR := src
SIM_DIR := sim
PY_DIR := python_test
BUILD_DIR := build
OBJ_DIR := obj_dir

# Files
VERILOG_TOP := $(SRC_DIR)/top.v
VERILOG_SRCS := $(wildcard $(SRC_DIR)/*.v)
CPP_SRCS := $(wildcard $(SIM_DIR)/*.cpp)

# Commands
VERILATOR := verilator
PYTHON := python3

# Verilator flags - updated for better compatibility
VERILATOR_FLAGS := -Wall --trace --cc --build
VERILATOR_INCLUDES := -I$(SRC_DIR) -I$(SIM_DIR)
CPP_FLAGS := -I$(OBJ_DIR)

# Make sure we can use C++17 features
CXXFLAGS := -std=c++17

# Default target
all: verilate train simulate compare

# Create directories
$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

# Verilate the design - updated command structure
verilate: | $(OBJ_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) $(VERILATOR_INCLUDES) $(VERILOG_TOP) $(CPP_SRCS)

# Train the model and export weights
train: | $(BUILD_DIR)
	mkdir -p $(PY_DIR)
	mkdir -p weights
	$(PYTHON) $(PY_DIR)/train_model.py

# Run Python baseline inference
python_inference:
	$(PYTHON) $(PY_DIR)/inference.py

# Run Verilog simulation
simulate: verilate
	./obj_dir/Vtop
	@echo "Verilog simulation complete."

# Compare results
compare:
	$(PYTHON) $(PY_DIR)/compare_results.py

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(OBJ_DIR)
	find $(PY_DIR) -name "*.bin" -delete || true
	find . -name "*.txt" -delete || true
	find . -name "*.vcd" -delete || true
	find . -name "*.png" -delete || true

.PHONY: all verilate train python_inference simulate compare clean