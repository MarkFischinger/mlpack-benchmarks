#include <iostream>
#include <chrono>
#include <omp.h>
#include <vector>
#include <complex>
#include <valarray>
#include <cmath>

// Direct Convolution function
void directConvolution(const std::vector<std::vector<int>>& input, const std::vector<std::vector<int>>& kernel, std::vector<std::vector<int>>& output) {
    int N = input.size();
    int M = kernel.size();
    int outputSize = N - M + 1;

    for (int i = 0; i < outputSize; ++i) {
        for (int j = 0; j < outputSize; ++j) {
            int sum = 0;
            for (int ki = 0; ki < M; ++ki) {
                for (int kj = 0; kj < M; ++kj) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
}

// Benchmark for Direct Convolution
void benchmarkDirectConvolution(int N, int M) {
    std::vector<std::vector<int>> input(N, std::vector<int>(N, 1));
    std::vector<std::vector<int>> kernel(M, std::vector<int>(M, 1));
    std::vector<std::vector<int>> output(N - M + 1, std::vector<int>(N - M + 1, 0));

    auto start = std::chrono::high_resolution_clock::now();
    directConvolution(input, kernel, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "Direct Convolution elapsed time: " << elapsed.count() << " ms\n";

    for (const auto& row : output) {
        for (const auto& element : row) {
            std::cout << element << ' ';
        }
        std::cout << '\n';
    }

}

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;

// Cooley–Tukey FFT (in-place, divide-and-conquer)
void fft(CArray& x, int start = 0, int step = 1) {
    const size_t N = x.size() / step;
    if (N <= 1) return;

    // conquer
    fft(x, start, step * 2);
    fft(x, start + step, step * 2);

    // combine
    for (size_t k = 0; k < N/2; ++k) {
        Complex t = std::polar(1.0, -2 * M_PI * k / N) * x[start + step * 2 * k + step];
        x[start + step * 2 * k + step] = x[start + step * 2 * k] - t;
        x[start + step * 2 * k] += t;
    }
}

// inverse FFT (in-place)
void ifft(CArray& x) {
    // conjugate the complex numbers
    for (auto& el : x) el = std::conj(el);

    // forward fft
    fft(x);

    // conjugate the complex numbers again and scale
    for (auto& el : x) el = std::conj(el) / static_cast<double>(x.size());
}

// FFT Convolution
void fftConvolution(std::vector<int>& input, std::vector<int>& kernel, std::vector<int>& output) {
    int N = input.size();
    int M = kernel.size();
    int size = N + M - 1;

    // Copy input vectors to complex arrays
    CArray data(size), resp(size);
    for (int i = 0; i < N; ++i) data[i] = input[i];
    for (int i = 0; i < M; ++i) resp[i] = kernel[i];

    // FFT
    fft(data);
    fft(resp);

    // Multiply pointwise
    for (int i = 0; i < size; ++i) data[i] *= resp[i];

    // Inverse FFT
    ifft(data);

    // Copy back to output vector
    for (int i = 0; i < size; ++i) output[i] = std::real(data[i]);
        std::cout << "FFT Convolution Output:\n";
        for (const auto& element : output) {
        std::cout << element << ' ';
    }
    std::cout << '\n';
}

// Benchmark for FFT Convolution
void benchmarkFFTConvolution(int N) {
    std::vector<int> input(N, 1);
    std::vector<int> kernel(N, 1);
    std::vector<int> output(N * 2 - 1, 0);

    auto start = std::chrono::high_resolution_clock::now();
    fftConvolution(input, kernel, output);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "FFT Convolution elapsed time: " << elapsed.count() << " ms\n";
}

int main() {
    int N = 9; // Input size
    int M = 3;   // Kernel size

    omp_set_num_threads(2);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();

        if (thread_id == 0) {
            benchmarkDirectConvolution(N, M);
        } else {
            benchmarkFFTConvolution(N);
        }
    }

    return 0;
}