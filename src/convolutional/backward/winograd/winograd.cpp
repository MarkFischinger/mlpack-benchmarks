#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

void winogradF2x2_3x3(const std::vector<std::vector<float>>& input,
                      const std::vector<std::vector<float>>& kernel,
                      std::vector<std::vector<float>>& output) {

    // Transformation matrices
    std::vector<std::vector<float>> G = {
        {1.0, 0.0, 0.0},
        {0.5, 0.5, 0.5},
        {0.5, -0.5, 0.5},
        {0.0, 0.0, 1.0}
    };

    std::vector<std::vector<float>> GT = {
        {1, 0.5, 0.5, 0},
        {0, 0.5, -0.5, 0},
        {0, 0.5, 0.5, 1}
    };

    std::vector<std::vector<float>> B = {
        {1, 0, -1, 0},
        {0, 1, 1, 0},
        {0, -1, 1, 0},
        {0, 1, 0, -1}
    };

    std::vector<std::vector<float>> BT = {
        {1, 0, 0, 0},
        {0, 1, -1, 1},
        {-1, 1, 1, 0},
        {0, 0, 0, -1}
    };

    int inputSize = input.size();
    int outputSize = inputSize - 2;


    std::vector<std::vector<float>> gKernel(3, std::vector<float>(3, 0));
    std::vector<std::vector<float>> U(4, std::vector<float>(4, 0));
  

    // Transform the kernel
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 3; ++k) {
                if (i < G.size() && k < G[i].size() && k < kernel.size() && j < kernel[k].size()) { 
                    U[i][j] += G[i][k] * kernel[k][j];
            }
        }
    }
}

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {

            // Extract tile
            std::vector<std::vector<float>> V(4, std::vector<float>(4, 0));
            for (int m = 0; m < 4; ++m) {
                for (int n = 0; n < 4; ++n) {
                    if (i + m < inputSize && j + n < inputSize) { 
                        V[m][n] = input[i + m][j + n];
                    }
                }
            }

            // Transform tile
            std::vector<std::vector<float>> Vt(4, std::vector<float>(4, 0));
            for (int m = 0; m < 4; ++m) {
                for (int n = 0; n < 4; ++n) {
                    for (int k = 0; k < 4; ++k) {
                        if (m < V.size() && k < V[m].size() && k < BT.size() && n < BT[k].size()) { 
                            Vt[m][n] += V[m][k] * BT[k][n];
                        }
                    }
                }
            }
            
            // Perform element-wise multiplication
            std::vector<std::vector<float>> M(4, std::vector<float>(4, 0));
            for (int m = 0; m < 4; ++m) {
                for (int n = 0; n < 4; ++n) {
                    if (m < Vt.size() && n < Vt[m].size() && m < U.size() && n < U[m].size()) { 
                        M[m][n] = Vt[m][n] * U[m][n];
                    }
                }
            }

            // Inverse transform
            for (int m = 0; m < 2; ++m) {
                for (int n = 0; n < 2; ++n) {
                    float sum = 0;
                    for (int k = 0; k < 4; ++k) {
                        if (m < BT.size() && k < BT[m].size() && k < M.size() && n < M[k].size()) { 
                            sum += BT[m][k] * M[k][n];
                        }
                    }
                    if (i + m < output.size() && j + n < output[i + m].size()) { 
                        output[i + m][j + n] = sum;
                    }
                }
            }
}
}
}

int main() {
    std::vector<std::vector<float>> input = {
        {1, 2, 3, 4, 5},
        {5, 6, 7, 8, 9},
        {9, 8, 7, 6, 5},
        {5, 4, 3, 2, 1},
        {1, 2, 3, 4, 5}
    };

    std::vector<std::vector<float>> kernel = {
        {0, 1, 2},
        {2, 2, 0},
        {0, 1, 2}
    };

    std::vector<std::vector<float>> output(3, std::vector<float>(3, 0));

    auto start = std::chrono::high_resolution_clock::now();
    winogradF2x2_3x3(input, kernel, output);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Winograd Convolution Output:" << std::endl;
    for (const auto& row : output) {
        for (float elem : row) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}