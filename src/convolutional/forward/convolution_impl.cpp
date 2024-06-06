#include <armadillo>
#include <chrono>
#include <iostream>
#include <fstream>

using namespace arma;

// apply padding to the input image
mat applyPadding(const mat& input, int padWidth, int padHeight, int inputWidth, int inputHeight, int inputDepth) {
    int paddedWidth = inputWidth + 2 * padWidth;
    int paddedHeight = inputHeight + 2 * padHeight;
    mat paddedInput = zeros<mat>(paddedWidth * paddedHeight * inputDepth, 1);

    for (int d = 0; d < inputDepth; d++) {
        for (int h = 0; h < inputHeight; h++) {
            for (int w = 0; w < inputWidth; w++) {
                int inputIndex = w + inputWidth * (h + inputHeight * d);
                int paddedIndex = (w + padWidth) + paddedWidth * ((h + padHeight) + paddedHeight * d);
                paddedInput(paddedIndex) = input(inputIndex);
            }
        }
    }
    return paddedInput;
}

// im2col function
mat im2col(const mat& inputPadded, int filterSize, int outputHeight, int outputWidth, int stride, int inputHeight, int inputWidth, int inputDepth) {
    int rows = filterSize * filterSize * inputDepth;
    int cols = outputHeight * outputWidth;
    mat columnizedInput(rows, cols);

    int col = 0;
    for (int h = 0; h < outputHeight; h++) {
        for (int w = 0; w < outputWidth; w++) {
            int row = 0;
            for (int d = 0; d < inputDepth; d++) {
                for (int fh = 0; fh < filterSize; fh++) {
                    for (int fw = 0; fw < filterSize; fw++) {
                        int hIndex = h * stride + fh;
                        int wIndex = w * stride + fw;
                        int inputIndex = wIndex + inputWidth * (hIndex + inputHeight * d);
                        columnizedInput(row++, col) = inputPadded(inputIndex);
                    }
                }
            }
            col++;
        }
    }
    return columnizedInput;
}

// im2col and GEMM
void forwardPass(const mat& input, mat& output, const mat& weights, int inputWidth, int inputHeight, int inputDepth, int filterSize, int stride, int padWidth, int padHeight) {
    int outputHeight = (inputHeight + 2 * padHeight - filterSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padWidth - filterSize) / stride + 1;

    mat inputPadded = applyPadding(input, padWidth, padHeight, inputWidth, inputHeight, inputDepth);

    mat columnizedInput = im2col(inputPadded, filterSize, outputHeight, outputWidth, stride, inputHeight + 2 * padHeight, inputWidth + 2 * padWidth, inputDepth);

    mat result = weights * columnizedInput;

    output = reshape(result, outputHeight, outputWidth);
}

void traditionalConvolution(const mat& input, mat& output, const mat& weights, int inputWidth, int inputHeight, int inputDepth, int filterSize, int stride, int padWidth, int padHeight) {
    int outputHeight = (inputHeight + 2 * padHeight - filterSize) / stride + 1;
    int outputWidth = (inputWidth + 2 * padWidth - filterSize) / stride + 1;

    mat inputPadded = applyPadding(input, padWidth, padHeight, inputWidth, inputHeight, inputDepth);

    output = zeros<mat>(outputHeight, outputWidth);

    for (int h = 0; h < outputHeight; h++) {
        for (int w = 0; w < outputWidth; w++) {
            for (int d = 0; d < inputDepth; d++) {
                for (int fh = 0; fh < filterSize; fh++) {
                    for (int fw = 0; fw < filterSize; fw++) {
                        int hIndex = h * stride + fh;
                        int wIndex = w * stride + fw;
                        int inputIndex = wIndex + inputWidth * (hIndex + inputHeight * d);
                        int weightIndex = fw + filterSize * (fh + filterSize * d);
                        output(h, w) += inputPadded(inputIndex) * weights(weightIndex);
                    }
                }
            }
        }
    }
}
int main() {
    std::vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048};  
    std::ofstream file("results.csv");
     file << "Size,ForwardPassTime,TraditionalConvolutionTime\n";

    for (int size : sizes) {
        int inputWidth = size;
        int inputHeight = size;
        int inputDepth = 3;
        int filterSize = 5;
        int stride = 1;
        int padWidth = 2;
        int padHeight = 2;
        mat input = randu<mat>(inputWidth * inputHeight * inputDepth, 1);  
        mat weights = randu<mat>(1, filterSize * filterSize * inputDepth); 
        mat output1, output2;

        auto start = std::chrono::high_resolution_clock::now();
        forwardPass(input, output1, weights, inputWidth, inputHeight, inputDepth, filterSize, stride, padWidth, padHeight);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end-start;
        double forwardPassTime = diff.count();

        start = std::chrono::high_resolution_clock::now();
        traditionalConvolution(input, output2, weights, inputWidth, inputHeight, inputDepth, filterSize, stride, padWidth, padHeight);
        end = std::chrono::high_resolution_clock::now();
        diff = end-start;
        double traditionalConvolutionTime = diff.count();

        file << size << "," << forwardPassTime << "," << traditionalConvolutionTime << "\n";


        if (output1.n_rows != output2.n_rows || output1.n_cols != output2.n_cols) {
            std::cout << "The output matrices have different dimensions.\n";
        } else if (approx_equal(output1, output2, "absdiff", 0.0001)) {
            std::cout << "The output matrices are approximately equal.\n";
        } else {
            std::cout << "The output matrices are not equal.\n";
            std::cout << "The difference is: " << accu(abs(output1 - output2)) << "\n";
        }
    }
file.close();

    return 0;
}