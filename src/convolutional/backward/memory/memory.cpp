/*
This file contains two functions, Convolution and OptimizedConvolution, which perform the convolution operation on input matrices. 
The Convolution function performs a basic convolution operation, while the OptimizedConvolution function uses OpenMP for parallelization and loop unrolling for optimization. 
The main function generates random input and filter matrices, then measures and compares the execution time of both convolution functions.
The logs from this code are in the "memory.log" file.
*/

#include <chrono>
#include <armadillo>
#include <iostream>

void Convolution(const arma::mat& input,
                 const arma::mat& filter,
                 arma::mat& output,
                 const size_t dW = 1,
                 const size_t dH = 1,
                 const size_t dilationW = 1,
                 const size_t dilationH = 1)
{
  typedef double eT;
  const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
  const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
  const size_t outputRows = (input.n_rows - filterRows + dH) / dH;
  const size_t outputCols = (input.n_cols - filterCols + dW) / dW;
  output.zeros(outputRows, outputCols);

  eT* outputPtr = output.memptr();

  for (size_t j = 0; j < output.n_cols; ++j)
  {
    for (size_t i = 0; i < output.n_rows; ++i, outputPtr++)
    {
      const eT* kernelPtr = filter.memptr();
      for (size_t kj = 0; kj < filter.n_cols; ++kj)
      {
        const eT* inputPtr = input.colptr(kj * dilationW + j * dW) + i * dH;
        for (size_t ki = 0; ki < filter.n_rows; ++ki, ++kernelPtr,
            inputPtr += dilationH)
          *outputPtr += *kernelPtr * (*inputPtr);
      }
    }
  }
}



void OptimizedConvolution(const arma::mat& input,
                          const arma::mat& filter,
                          arma::mat& output,
                          const size_t dW = 1,
                          const size_t dH = 1,
                          const size_t dilationW = 1,
                          const size_t dilationH = 1)
{
    typedef double eT;
    const size_t filterRows = filter.n_rows * dilationH - (dilationH - 1);
    const size_t filterCols = filter.n_cols * dilationW - (dilationW - 1);
    const size_t outputRows = (input.n_rows - filterRows + dH) / dH;
    const size_t outputCols = (input.n_cols - filterCols + dW) / dW;
    output.zeros(outputRows, outputCols);

    eT* outputPtr = output.memptr();

    #pragma omp parallel for
    for (size_t j = 0; j < output.n_cols; ++j)
    {
        for (size_t i = 0; i < output.n_rows; ++i, outputPtr++)
        {
            const eT* kernelPtr = filter.memptr();
            for (size_t kj = 0; kj < filter.n_cols; ++kj)
            {
                const eT* inputPtr = input.colptr(kj * dilationW + j * dW) + i * dH;
                for (size_t ki = 0; ki < filter.n_rows; ki += 4, kernelPtr += 4, inputPtr += 4 * dilationH)
                {
                    *outputPtr += kernelPtr[0] * inputPtr[0];
                    *outputPtr += kernelPtr[1] * inputPtr[1];
                    *outputPtr += kernelPtr[2] * inputPtr[2];
                    *outputPtr += kernelPtr[3] * inputPtr[3];
                }
            }
        }
    }
}

int main()
{
    arma::mat input = arma::randu<arma::mat>(10000, 10000);
    arma::mat filter = arma::randu<arma::mat>(3, 3);
    arma::mat output1, output2;

    const int numRuns = 10; 
    double totalTime1 = 0, totalTime2 = 0;

    for (int run = 0; run < numRuns; ++run)
    {
        auto start1 = std::chrono::high_resolution_clock::now();
        Convolution(input, filter, output1);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = end1-start1;
        totalTime1 += diff1.count();

        auto start2 = std::chrono::high_resolution_clock::now();
        OptimizedConvolution(input, filter, output2);
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff2 = end2-start2;
        totalTime2 += diff2.count();
    }

    std::cout << "Average time to compute convolution with original function: " << totalTime1 / numRuns << " s\n";
    std::cout << "Average time to compute convolution with optimized function: " << totalTime2 / numRuns << " s\n";

    return 0;
}