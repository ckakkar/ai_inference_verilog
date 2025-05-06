// dense_layer.v
// Fully Connected Layer for Neural Network Inference
// Implements matrix-vector multiplication

module dense_layer #(
    parameter INPUT_SIZE = 784,
    parameter OUTPUT_SIZE = 10,
    parameter WEIGHT_FILE = "weights/dense_weights.txt"
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [7:0] in_data [0:INPUT_SIZE-1],
    output reg [31:0] out_data [0:OUTPUT_SIZE-1],
    output reg done
);

    // Weight memory - stores weights as 8-bit signed values
    reg [7:0] weights [0:OUTPUT_SIZE-1][0:INPUT_SIZE-1];
    
    // State machine states
    localparam IDLE = 2'b00;
    localparam COMPUTE = 2'b01;
    localparam DONE = 2'b10;
    localparam INVALID = 2'b11; // Add this to complete case statement
    
    reg [1:0] state;
    reg [31:0] accumulators [0:OUTPUT_SIZE-1];
    reg [15:0] input_idx;
    reg [15:0] output_idx;
    
    // Initialize weights from file
    initial begin
        $readmemh(WEIGHT_FILE, weights);
    end
    
    integer i; // Add this for the for loop
    
    // State machine for matrix-vector multiplication
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            input_idx <= 0;
            output_idx <= 0;
            for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                accumulators[i] <= 32'h0;
                out_data[i] <= 32'h0;
            end
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= COMPUTE;
                        input_idx <= 0;
                        output_idx <= 0;
                        done <= 1'b0;
                        for (i = 0; i < OUTPUT_SIZE; i = i + 1) begin
                            accumulators[i] <= 32'h0;
                        end
                    end
                end
                
                COMPUTE: begin
                    // Compute one neuron at a time
                    if (input_idx < INPUT_SIZE) begin
                        accumulators[output_idx] <= accumulators[output_idx] + 
                                                 $signed(in_data[input_idx]) * 
                                                 $signed(weights[output_idx][input_idx]);
                        input_idx <= input_idx + 1;
                    end else begin
                        // Move to next neuron
                        out_data[output_idx] <= accumulators[output_idx];
                        output_idx <= output_idx + 1;
                        input_idx <= 0;
                        
                        // Check if all neurons are computed
                        if (output_idx == OUTPUT_SIZE - 1) begin
                            state <= DONE;
                        end
                    end
                end
                
                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                
                INVALID: begin
                    // Handle invalid state
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule