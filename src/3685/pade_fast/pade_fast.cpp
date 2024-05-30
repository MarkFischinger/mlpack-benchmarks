#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h> 
#include <omp.h> 
#include <cmath> 

#define AVGITER 10000000

double lambda1(double x) {
    static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
    if (x < 4.0) {
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y; y *= y; y *= y;
        return 1 / y;
    }
    return 0.0;
}

double lambda2(double x) {
    static constexpr double num_coeffs[] = {120, -60, 12}, den_coeffs[] = {120, 60, 12};
    double num = num_coeffs[0] + x * (num_coeffs[1] + x * num_coeffs[2]);
    double den = den_coeffs[0] + x * (den_coeffs[1] + x * den_coeffs[2]);
    return num / den;
}

int main() {
    uint64_t st = 0, et = 0, avg1 = 0, avg2 = 0;
    double error1 = 0, error2 = 0;
    volatile double result = 0;

    omp_set_num_threads(4); 

    #pragma omp parallel for reduction(+:avg1, avg2, error1, error2, result)
    for(int i = 0; i < AVGITER; i++) {
        double x = rand() % 2; 
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