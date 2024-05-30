#include <iostream>
#include <vector>
#include <omp.h>
#include <armadillo>

using namespace arma;

template<typename InMatType, typename FilMatType, typename OutMatType>
void Convolution(const InMatType& input,
                 const FilMatType& filter,
                 OutMatType& output,
                 const size_t dW = 1,
                 const size_t dH = 1,
                 const size_t dilationW = 1,
                 const size_t dilationH = 1,
                 const bool appending = false)
{
    typedef typename InMatType::elem_type eT;
    if (!appending)
    {
        const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
        const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
        const size_t outputRows = (input.n_rows - filterRows + dH) / dH;
        const size_t outputCols = (input.n_cols - filterCols + dW) / dW;
        output.zeros(outputRows, outputCols);
    }

    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (size_t j = 0; j < output.n_cols; ++j)
        {
            for (size_t i = 0; i < output.n_rows; ++i)
            {
                eT* outputPtr = &output(i, j);
                const eT* kernelPtr = filter.memptr();
                for (size_t kj = 0; kj < filter.n_cols; ++kj)
                {
                    const eT* inputPtr = input.colptr(kj * dilationW + j * dW) + i * dH;
                    for (size_t ki = 0; ki < filter.n_rows; ++ki, ++kernelPtr, inputPtr += dilationH)
                    {
                        #pragma omp atomic
                        *outputPtr += *kernelPtr * (*inputPtr);
                    }
                }
            }
        }
    }
}

const int N = 1024; // Size of the input matrix (NxN)
const int M = 256;  // Size of the filter matrix (MxM)
const int C = 1024; // Output size (CxN)

void winogradAlgorithm(const std::vector<std::vector<float>>& input,
                       const std::vector<std::vector<float>>& filter,
                       std::vector<std::vector<float>>& output) {
    #pragma omp parallel
    {
        #pragma omp for collapse(2)
        for (int bx = 0; bx < N; bx += 16) {
            for (int by = 0; by < N; by += 16) {
                float local_acc[16][16] = {0}; // Local accumulator for the results

                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        float accumulator = 0.0;
                        for (int k = 0; k < M; k++) {
                            accumulator += input[bx + i][k] * filter[k][by + j];
                        }
                        local_acc[i][j] = accumulator;
                    }
                }

                // Save the results from local accumulator to the output matrix
                for (int i = 0; i < 16; i++) {
                    for (int j = 0; j < 16; j++) {
                        #pragma omp critical
                        output[bx + i][by + j] += local_acc[i][j];
                    }
                }
            }
        }
    }
}

int main() {
    std::vector<std::vector<float>> input(N, std::vector<float>(N, 1.0)); 
    std::vector<std::vector<float>> filter(M, std::vector<float>(M, 1.0)); 
    std::vector<std::vector<float>> output(N, std::vector<float>(N, 0.0));

    double start_time = omp_get_wtime();
    winogradAlgorithm(input, filter, output);
    double end_time = omp_get_wtime();

    std::cout << "Execution time: " << (end_time - start_time) << " seconds" << std::endl;

    return 0;
}
