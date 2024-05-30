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