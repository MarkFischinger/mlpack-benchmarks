#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <omp.h>
#include <cmath>

#define AVGITER 10000000

// Chebyshev coefficients for exp(-x) on [0, 13], degree 6 
static const double chebCoeffs[] = {1.00000, -0.88759, 0.37612, -0.08223, 0.01056, -0.00074, 0.00002};

double lambda1(double x) {
    static constexpr double A0 = 1.0, A1 = 0.125, A2 = 0.0078125, A3 = 0.00032552083, A4 = 1.0172526e-5;
    if (x < 4.0) {
        double y = A0 + x * (A1 + x * (A2 + x * (A3 + x * A4)));
        y *= y; y *= y; y *= y;
        return 1 / y;
    }
    return 0.0;
}

float enhancedLambda1(float x) {
    if (x < -126) return 0.0f;
    else if (x > 128) return INFINITY;

    alignas(16) static const float ct[8] = {
        1.44269502f,  // lb(e)
        1.92596299E-8f, // Correction to the value lb(e)
        -0.000921120925f, // 16*b2
        0.115524396f, // 4*b1
        2.88539004f, // b0
        2.0f, // 2
        4.65661287E-10f, // 2^-31
        0.0f  // Padding for alignment
    };

    __m128 x_vec = _mm_set_ss(x);
    __m128 ct_vec = _mm_load_ps(ct);

    __m128 t_vec = _mm_mul_ss(x_vec, ct_vec);
    int k = _mm_cvtss_si32(t_vec);

    if (x < 0) k += 32;

    __m128 k_vec = _mm_cvtsi32_ss(_mm_setzero_ps(), k);
    t_vec = _mm_fmsub_ss(x_vec, ct_vec, k_vec);

    __m128 t2_vec = _mm_mul_ss(t_vec, t_vec);
    __m128 b2_vec = _mm_load_ss(&ct[2]);
    __m128 b1_vec = _mm_load_ss(&ct[3]);
    __m128 b0_vec = _mm_load_ss(&ct[4]);
    __m128 result = _mm_fmadd_ss(b2_vec, t2_vec, b1_vec);
    result = _mm_fmadd_ss(result, t2_vec, b0_vec);
    result = _mm_fmadd_ss(result, t_vec, b0_vec);

    float final_result = std::ldexp(_mm_cvtss_f32(result), k - 127);

    return final_result;
}

int main() {
    uint64_t st = 0, et = 0, avgLambda1 = 0, avgEnhanced = 0;
    double errorLambda1 = 0, errorEnhanced = 0;
    volatile double resultLambda1 = 0, resultEnhanced = 0; 

    omp_set_num_threads(4); 

    // Benchmarking and error calculation
    #pragma omp parallel for reduction(+:avgLambda1, avgEnhanced, errorLambda1, errorEnhanced, resultLambda1, resultEnhanced)
    for(int i = 0; i < AVGITER; i++) {
        double x = static_cast<double>(rand() % 14); 
        double accurate = exp(-x);

        st = __rdtsc();
        double resLambda1 = lambda1(x);
        et = __rdtsc();
        avgLambda1 += et - st;
        errorLambda1 += fabs(resLambda1 - accurate) / accurate; 

        st = __rdtsc();
        double resEnhanced = enhancedLambda1(x);
        et = __rdtsc();
        avgEnhanced += et - st;
        errorEnhanced += fabs(resEnhanced - accurate) / accurate; 

        resultLambda1 += resLambda1; 
        resultEnhanced += resEnhanced; 
    }

    printf("Lambda1 Cycles: %'lu, Average Relative Error: %f\n", avgLambda1 / AVGITER, errorLambda1 / AVGITER);
    printf("Enhanced Lambda1 Cycles: %'lu, Average Relative Error: %f\n", avgEnhanced / AVGITER, errorEnhanced / AVGITER);
    printf("Dummy output to keep computations: Lambda1: %f, Enhanced: %f\n", resultLambda1, resultEnhanced);

    return 0;
}