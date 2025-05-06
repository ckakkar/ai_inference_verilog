// relu_activation.v
// ReLU Activation Function for Neural Network Inference
// f(x) = max(0, x)

module relu_activation (
    input wire clk,
    input wire rst_n,
    input wire [31:0] in_data,
    input wire valid_in,
    output reg [31:0] out_data,
    output reg valid_out
);

    // ReLU logic - output is max(0, input)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_data <= 32'h0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            // ReLU function: max(0, x)
            out_data <= ($signed(in_data) > 0) ? in_data : 32'h0;
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule