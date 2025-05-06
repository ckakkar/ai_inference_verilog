// main.cpp
// Verilator testbench for Neural Network Inference Accelerator

#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
// Update include paths to look in obj_dir
#include "../obj_dir/Vtop.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

// Configuration parameters
const int INPUT_SIZE = 784;   // MNIST input size (28x28)
const int OUTPUT_SIZE = 10;   // Number of output classes (0-9 digits)
const int NUM_TESTS = 100;    // Number of test images to process

// Function to load MNIST test images (simplified format)
std::vector<std::vector<uint8_t>> loadMnistImages(const std::string& filename, int numImages) {
    std::vector<std::vector<uint8_t>> images;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return images;
    }
    
    // Read images
    for (int i = 0; i < numImages; ++i) {
        std::vector<uint8_t> image(INPUT_SIZE);
        file.read(reinterpret_cast<char*>(image.data()), INPUT_SIZE);
        images.push_back(image);
    }
    
    return images;
}

// Function to load MNIST test labels
std::vector<uint8_t> loadMnistLabels(const std::string& filename, int numLabels) {
    std::vector<uint8_t> labels;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return labels;
    }
    
    // Read labels
    labels.resize(numLabels);
    file.read(reinterpret_cast<char*>(labels.data()), numLabels);
    
    return labels;
}

// Function to save inference results
void saveResults(const std::string& filename, const std::vector<std::vector<int32_t>>& results) {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }
    
    for (const auto& result : results) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            file << result[i] << " ";
        }
        file << std::endl;
    }
}

int main(int argc, char** argv) {
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    
    // Create Verilator model
    Vtop* top = new Vtop;
    
    // Initialize VCD trace dump
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("trace.vcd");
    
    // Load test data
    std::vector<std::vector<uint8_t>> testImages = loadMnistImages("python_test/mnist_test.bin", NUM_TESTS);
    std::vector<uint8_t> testLabels = loadMnistLabels("python_test/mnist_labels.bin", NUM_TESTS);
    
    if (testImages.empty() || testLabels.empty()) {
        std::cerr << "Error: Could not load test data" << std::endl;
        return 1;
    }
    
    // Storage for results
    std::vector<std::vector<int32_t>> results;
    std::vector<int> predictions;
    
    // Clock cycle counter
    uint64_t cycleCount = 0;
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Process each test image
    for (int imgIdx = 0; imgIdx < NUM_TESTS; ++imgIdx) {
        const auto& image = testImages[imgIdx];
        
        // Reset the model
        top->rst_n = 0;
        top->clk = 0;
        top->eval();
        tfp->dump(cycleCount++);
        top->clk = 1;
        top->eval();
        tfp->dump(cycleCount++);
        top->rst_n = 1;
        
        // Load input data
        for (int i = 0; i < INPUT_SIZE; ++i) {
            top->in_data[i] = image[i];
        }
        
        // Start inference
        top->start = 1;
        top->clk = 0;
        top->eval();
        tfp->dump(cycleCount++);
        top->clk = 1;
        top->eval();
        tfp->dump(cycleCount++);
        top->start = 0;
        
        // Run until done
        while (!top->done) {
            top->clk = 0;
            top->eval();
            tfp->dump(cycleCount++);
            top->clk = 1;
            top->eval();
            tfp->dump(cycleCount++);
        }
        
        // Capture output
        std::vector<int32_t> result(OUTPUT_SIZE);
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            result[i] = top->out_data[i];
        }
        results.push_back(result);
        
        // Determine prediction (index of max value)
        int prediction = 0;
        int32_t maxVal = result[0];
        for (int i = 1; i < OUTPUT_SIZE; ++i) {
            if (result[i] > maxVal) {
                maxVal = result[i];
                prediction = i;
            }
        }
        predictions.push_back(prediction);
        
        std::cout << "Image " << imgIdx << ": Predicted " << prediction 
                  << ", Actual " << static_cast<int>(testLabels[imgIdx]) << std::endl;
    }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    // Calculate accuracy
    int correctCount = 0;
    for (int i = 0; i < NUM_TESTS; ++i) {
        if (predictions[i] == testLabels[i]) {
            correctCount++;
        }
    }
    double accuracy = static_cast<double>(correctCount) / NUM_TESTS * 100.0;
    
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "Inference time: " << duration << " ms for " << NUM_TESTS << " images" << std::endl;
    std::cout << "Average time per image: " << static_cast<double>(duration) / NUM_TESTS << " ms" << std::endl;
    std::cout << "Clock cycles: " << cycleCount << std::endl;
    
    // Save results
    saveResults("verilog_results.txt", results);
    std::ofstream timeFile("verilog_time.txt");
    if (timeFile.is_open()) {
        timeFile << static_cast<double>(duration) / 1000.0; // Convert to seconds
        timeFile.close();
    }
    
    // Cleanup
    tfp->close();
    delete tfp;
    delete top;
    
    return 0;
}