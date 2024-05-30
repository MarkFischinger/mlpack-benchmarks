#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <x86intrin.h> 
#include <omp.h> 
#include <vector>
#include <armadillo> 

#define AVGITER 100

void benchmark_mlpack_time(const arma::mat& input, double ratio);
void benchmark_omar_time(arma::mat& input, double ratio);
void benchmark_find_and_fill_time(arma::mat& input, double ratio);
void benchmark_conv_to_mat_time(const arma::mat& input, double ratio);

int main() {
    omp_set_num_threads(4); 

    std::vector<int> sizes = {10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    double ratio = 0.5; 

    for (int size : sizes) {
        arma::mat input = arma::randu<arma::mat>(size, size); 

        printf("\n\nBenchmarking for size: %d\n", size);

        benchmark_mlpack_time(input, ratio);
        benchmark_omar_time(input, ratio);
        benchmark_find_and_fill_time(input, ratio);
        benchmark_conv_to_mat_time(input, ratio);
    }

    return 0;
}

void benchmark_mlpack_time(const arma::mat& input, double ratio) {
    uint64_t st = 0, et = 0, avg = 0;
    arma::mat output = input;

    #pragma omp parallel for reduction(+:avg)
    for(int i = 0; i < AVGITER; i++) {
        st = __rdtsc();
        output.transform([&] (double val) { return (val > ratio); });
        et = __rdtsc();
        avg += et - st;
    }
    printf("mlpack_time Cycles: %'lu\n", avg / AVGITER);
}

void benchmark_omar_time(arma::mat& input, double ratio) {
    uint64_t st = 0, et = 0, avg = 0;
    arma::mat output_2 = input;

    #pragma omp parallel for reduction(+:avg)
    for(int i = 0; i < AVGITER; i++) {
        st = __rdtsc();
        output_2 = arma::sign(output_2 - ratio);
        output_2.clamp(0, 1);
        et = __rdtsc();
        avg += et - st;
    }
    printf("omar_time Cycles: %'lu\n", avg / AVGITER);
}

void benchmark_find_and_fill_time(arma::mat& input, double ratio) {
    uint64_t st = 0, et = 0, avg = 0;
    arma::mat output_3;

    #pragma omp parallel for private(output_3) reduction(+:avg)
    for(int i = 0; i < AVGITER; i++) {
        output_3 = input; // Ensure each thread has its own copy
        st = __rdtsc();
        output_3.transform([&ratio](double val) -> double { return val > ratio ? 1.0 : 0.0; });
        et = __rdtsc();
        avg += et - st;
    }
    printf("find_and_fill_time Cycles: %'lu\n", avg / AVGITER);
}

void benchmark_conv_to_mat_time(const arma::mat& input, double ratio) {
    uint64_t st = 0, et = 0, avg = 0;

    #pragma omp parallel
    {
        arma::mat local_output = input; // Each thread gets its own copy
        uint64_t local_avg = 0;

        #pragma omp for
        for(int i = 0; i < AVGITER; i++) {
            st = __rdtsc();
            local_output = arma::conv_to<arma::mat>::from(local_output > ratio);
            et = __rdtsc();
            local_avg += et - st;
        }

        #pragma omp critical
        avg += local_avg;
    }
    printf("conv_to_mat_time Cycles: %'lu\n", avg / AVGITER);
}