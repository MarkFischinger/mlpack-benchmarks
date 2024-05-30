#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <x86intrin.h> 
#include <omp.h> 
#include <cmath> 
#include <iostream>

#define AVGITER 10000000

double lambda1(double x) {
    if (x < 13.0) {
        static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y; y *= y; y *= y;
        return 1 / y;
    }
    return 0.0;
}

double lambda2(double x) {
            if (x >= 13.0) {
                return 0.0;
            }

            static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
            double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
            y *= y; y *= y; y *= y;
            return 1 / y;
}

int main() {
    uint64_t st = 0, et = 0, avg1 = 0, avg2 = 0;
    double error1 = 0, error2 = 0;
    volatile double result = 0; 

    omp_set_num_threads(4); 

    #pragma omp parallel for reduction(+:avg1, avg2, error1, error2, result)
    for(int i = 0; i < AVGITER; i++) {
        double x = rand() % 13; 
        double accurate = exp(-x);

        st = __rdtsc();
        double res1 = lambda1(x);
        et = __rdtsc();
        avg1 += et - st;
        error1 += fabs(res1 - accurate) / accurate; 

        st = __rdtsc();
        double res2 = lambda2(x);
        et = __rdtsc();
        avg2 += et - st;
        error2 += fabs(res2 - accurate) / accurate; 

        result += res1 + res2; 
    }

    printf("Lambda1 Cycles: %'lu, Average Relative Error: %f\n", avg1 / AVGITER, error1 / AVGITER);
    printf("Lambda2 Cycles: %'lu, Average Relative Error: %f\n", avg2 / AVGITER, error2 / AVGITER);
    printf("Dummy output to keep computations: %f\n", result);

    return 0;
}





#include <iostream>
#include <chrono>
#include <omp.h>

void algorithmA() {
    for (volatile int i = 0; i < 100000000; ++i) {}
}

void algorithmB() {
    for (volatile int i = 0; i < 100000000; ++i) {}
}

int main() {
    omp_set_num_threads(2); 

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        omp_set_num_threads(2);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(thread_id, &cpuset); 
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        auto start = std::chrono::high_resolution_clock::now();

        if (thread_id == 0) {
            algorithmA();
        } else {
            algorithmB();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        #pragma omp critical
        {
            std::cout << "Thread " << thread_id << " elapsed time: " << elapsed.count() << " ms\n";
        }
    }

    return 0;
}
