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

# Verilator flags
VERILATOR_FLAGS := -Wall --trace -cc --exe
VERILATOR_INCLUDES := -I$(SRC_DIR) -I$(SIM_DIR)

# Default target
all: verilate train simulate compare

# Create directories
$(BUILD_DIR) $(OBJ_DIR):
	mkdir -p $@

# Verilate the design
verilate: $(OBJ_DIR)/Vtop
$(OBJ_DIR)/Vtop: $(VERILOG_SRCS) $(CPP_SRCS) | $(OBJ_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) $(VERILATOR_INCLUDES) $(VERILOG_TOP) $(CPP_SRCS) -o Vtop
	make -C $(OBJ_DIR) -f Vtop.mk Vtop

# Train the model and export weights
train: | $(BUILD_DIR)
	mkdir -p $(PY_DIR)
	$(PYTHON) $(PY_DIR)/train_model.py

# Run Python baseline inference
python_inference:
	$(PYTHON) $(PY_DIR)/inference.py

# Run Verilog simulation
simulate: $(OBJ_DIR)/Vtop
	cd $(OBJ_DIR) && ./Vtop
	@echo "Verilog simulation complete."

# Compare results
compare:
	$(PYTHON) $(PY_DIR)/compare_results.py

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(OBJ_DIR)
	find $(PY_DIR) -name "*.bin" -delete
	find . -name "*.txt" -delete
	find . -name "*.vcd" -delete
	find . -name "*.png" -delete

.PHONY: all verilate train python_inference simulate compare clean