#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> 
#include <x86intrin.h> 
#include <omp.h> 
#include <cmath> 

#define AVGITER 100000000

// Scaled Pade Approximation for exp(-x)
auto scaledPadeApproxExpMinusX = [](double x) {
    if (x < 13.0) {
        double s = 4.0;

        double xs = x / s;

        double numerator = 24 - 12*xs + 4*xs*xs - xs*xs*xs;
        double denominator = 24 + 12*xs + 4*xs*xs + xs*xs*xs;

        double pade = numerator / denominator;
        return std::pow(pade, s);
    }

    return 0.0;
};

double lambda1(double x) {
    static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
    if (x < 13.0) {
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y; y *= y; y *= y;
        return 1 / y;
    }
    return 0.0;
};

double combinedApprox(double x) {
    if (x < 4.0) {
        static constexpr double num_coeffs[] = {120, -60, 12};
        static constexpr double den_coeffs[] = {120, 60, 12};

        double num = num_coeffs[0] + x * (num_coeffs[1] + x * num_coeffs[2]);
        double den = den_coeffs[0] + x * (den_coeffs[1] + x * den_coeffs[2]);

        return num / den;
    } else if (x < 13.0) {
        static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y; y *= y; y *= y;
        return 1 / y;
    }
    return 0.0;
};

int main() {
    uint64_t st = 0, et = 0, avgScaled = 0, avgLambda1 = 0, avgCombined = 0;
    double errorScaled = 0, errorLambda1 = 0, errorCombined = 0;
    volatile double resultScaled = 0, resultLambda1 = 0, resultCombined = 0;

    omp_set_num_threads(4); 

    #pragma omp parallel for reduction(+:avgScaled, errorScaled, resultScaled, avgLambda1, errorLambda1, resultLambda1, avgCombined, errorCombined, resultCombined)
    for(int i = 0; i < AVGITER; i++) {
        double x = rand() % 13; 
        double accurate = exp(-x);

        st = __rdtsc();
        double resScaled = scaledPadeApproxExpMinusX(x);
        et = __rdtsc();
        avgScaled += et - st;
        errorScaled += fabs(resScaled - accurate) / accurate; 
        resultScaled += resScaled; 

        st = __rdtsc();
        double resLambda1 = lambda1(x);
        et = __rdtsc();
        avgLambda1 += et - st;
        errorLambda1 += fabs(resLambda1 - accurate) / accurate; 
        resultLambda1 += resLambda1; 

        st = __rdtsc();
        double resCombined = combinedApprox(x);
        et = __rdtsc();
        avgCombined += et - st;
        errorCombined += fabs(resCombined - accurate) / accurate; 
        resultCombined += resCombined; 
    }

    printf("Scaled Pade Approximation Cycles: %'lu, Average Relative Error: %f\n", avgScaled / AVGITER, errorScaled / AVGITER);
    printf("Lambda1 Cycles: %'lu, Average Relative Error: %f\n", avgLambda1 / AVGITER, errorLambda1 / AVGITER);
    printf("Combined Approximation Cycles: %'lu, Average Relative Error: %f\n", avgCombined / AVGITER, errorCombined / AVGITER);
    printf("Dummy output to keep computations: Scaled: %f, Lambda1: %f, Combined: %f\n", resultScaled, resultLambda1, resultCombined);

    return 0;
}