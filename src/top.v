// top.v
// Top module for Neural Network Inference Accelerator
// Includes a dense layer with ReLU activation

module top #(
    parameter INPUT_SIZE = 784,   // MNIST input size (28x28)
    parameter HIDDEN_SIZE = 16,   // Hidden layer size
    parameter OUTPUT_SIZE = 10,   // Number of output classes (0-9 digits)
    parameter HIDDEN_WEIGHT_FILE = "weights/hidden_weights.txt",
    parameter OUTPUT_WEIGHT_FILE = "weights/output_weights.txt"
) (
    input wire clk,
    input wire rst_n,
    input wire start,
    input wire [7:0] in_data [0:INPUT_SIZE-1],
    output reg [31:0] out_data [0:OUTPUT_SIZE-1],
    output reg done
);

    // Intermediate signals
    wire [31:0] hidden_out [0:HIDDEN_SIZE-1];
    reg [31:0] relu_out [0:HIDDEN_SIZE-1];
    reg hidden_done;
    wire relu_done; // Changed from reg to wire
    reg output_done;
    
    // State machine states
    localparam IDLE = 2'b00;
    localparam PROCESS_HIDDEN = 2'b01;
    localparam PROCESS_OUTPUT = 2'b10;
    localparam DONE = 2'b11;
    
    reg [1:0] state;
    reg hidden_start, output_start;
    
    // Dense layer for hidden layer
    dense_layer #(
        .INPUT_SIZE(INPUT_SIZE),
        .OUTPUT_SIZE(HIDDEN_SIZE),
        .WEIGHT_FILE(HIDDEN_WEIGHT_FILE)
    ) hidden_layer (
        .clk(clk),
        .rst_n(rst_n),
        .start(hidden_start),
        .in_data(in_data),
        .out_data(hidden_out),
        .done(hidden_done)
    );
    
    // ReLU activation for hidden layer outputs
    // Changed the generate block to avoid array indexing issues
    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                relu_out[i] <= 32'h0;
            end
        end else if (hidden_done) begin
            for (i = 0; i < HIDDEN_SIZE; i = i + 1) begin
                relu_out[i] <= (hidden_out[i][31]) ? 32'h0 : hidden_out[i]; // ReLU function
            end
        end
    end
    
    // Generate relu_done signal
    reg relu_done_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            relu_done_reg <= 1'b0;
        else
            relu_done_reg <= hidden_done; // One cycle delay
    end
    assign relu_done = relu_done_reg;
    
    // Dense layer for output layer
    dense_layer #(
        .INPUT_SIZE(HIDDEN_SIZE),
        .OUTPUT_SIZE(OUTPUT_SIZE),
        .WEIGHT_FILE(OUTPUT_WEIGHT_FILE)
    ) output_layer (
        .clk(clk),
        .rst_n(rst_n),
        .start(output_start),
        .in_data(relu_out),
        .out_data(out_data),
        .done(output_done)
    );
    
    // State machine to control network execution
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            hidden_start <= 1'b0;
            output_start <= 1'b0;
            done <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= PROCESS_HIDDEN;
                        hidden_start <= 1'b1;
                        done <= 1'b0;
                    end
                end
                
                PROCESS_HIDDEN: begin
                    hidden_start <= 1'b0;
                    if (hidden_done) begin
                        state <= PROCESS_OUTPUT;
                        output_start <= 1'b1;
                    end
                end
                
                PROCESS_OUTPUT: begin
                    output_start <= 1'b0;
                    if (output_done) begin
                        state <= DONE;
                    end
                end
                
                DONE: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                
                default: state <= IDLE; // Add default case
            endcase
        end
    end

endmodule