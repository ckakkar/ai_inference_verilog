// mac_unit.v
// Multiply-Accumulate Unit for Neural Network Inference
// Input: 8-bit signed values, Output: 32-bit accumulated result

module mac_unit (
    input wire clk,
    input wire rst_n,
    input wire [7:0] in_data,
    input wire [7:0] weight,
    input wire valid_in,
    output reg [31:0] acc_out,
    output reg valid_out
);

    // Product of input and weight
    wire [15:0] product;
    assign product = $signed(in_data) * $signed(weight);
    
    // Accumulate logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 32'h0;
            valid_out <= 1'b0;
        end else if (valid_in) begin
            acc_out <= acc_out + $signed(product);
            valid_out <= 1'b1;
        end else begin
            valid_out <= 1'b0;
        end
    end

endmodule