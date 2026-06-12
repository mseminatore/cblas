//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 8192

// Wall-clock seconds. NOTE: do NOT use the library's cbu_timer here — on POSIX it
// uses CLOCK_PROCESS_CPUTIME_ID (summed CPU time across all threads), which
// underreports multi-threaded throughput by ~the thread count. GFLOPS must be
// measured against elapsed wall time, matching the OpenBLAS comparison bench.
// Portable: QueryPerformanceCounter on Windows (MSVC has no clock_gettime /
// CLOCK_MONOTONIC), CLOCK_MONOTONIC elsewhere. <windows.h> arrives via cblas.h.
static double bench_now(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_gemm(void)
{
    CBLAS_INDEX m, n, k;
    float dt;
    
    // Allocate matrices on heap to avoid stack overflow
    float *a = (float*)malloc(MAX_SIZE * MAX_SIZE * sizeof(float));
    float *b = (float*)malloc(MAX_SIZE * MAX_SIZE * sizeof(float));
    float *c = (float*)malloc(MAX_SIZE * MAX_SIZE * sizeof(float));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Failed to allocate matrices (need %zu MB each)\\n", 
                (size_t)(MAX_SIZE * MAX_SIZE * sizeof(float)) / (1024*1024));
        if (a) fprintf(stderr, "  a: allocated\\n");
        if (b) fprintf(stderr, "  b: allocated\\n");
        if (c) fprintf(stderr, "  c: allocated\\n");
        free(a); free(b); free(c);
        return;
    }
    
    printf("Allocated 3 matrices of %dx%d floats (%zu MB total)\n\n", 
           MAX_SIZE, MAX_SIZE, (size_t)(3 * MAX_SIZE * MAX_SIZE * sizeof(float)) / (1024*1024));
    
    // Initialize matrices
    memset(a, 0, MAX_SIZE * MAX_SIZE * sizeof(float));
    memset(b, 0, MAX_SIZE * MAX_SIZE * sizeof(float));
    memset(c, 0, MAX_SIZE * MAX_SIZE * sizeof(float));

    printf("Testing performance of cblas_sgemm()\n\n");

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = k = i;
        
        printf("Testing size %d...", i);
        fflush(stdout);

        // For small sizes, run multiple iterations to get stable timing
        int iters = 1;
        if (i <= 512) iters = 10;
        if (i <= 128) iters = 100;
        if (i <= 32) iters = 1000;
        
        double w1 = bench_now();

        for (int iter = 0; iter < iters; iter++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, MAX_SIZE, b, MAX_SIZE, 1.0f, c, MAX_SIZE);
        }

        dt = (float)((bench_now() - w1) / iters);

        printf(" %5.2f GFlops in %5.2fs\n", (float)2 * m * n * k / 1000000000 / dt, dt);
    }
    
    // Also test with contiguous matrices (lda = n) to show cache-friendly case
    printf("\nTesting with contiguous layout (lda=n):\n\n");
    for (int i = 128; i <= 1024; i <<= 1)
    {
        m = n = k = i;
        
        printf("Testing size %d...", i);
        fflush(stdout);

        // Allocate contiguous matrices
        float *a_cont = (float*)malloc((size_t)i * i * sizeof(float));
        float *b_cont = (float*)malloc((size_t)i * i * sizeof(float));
        float *c_cont = (float*)malloc((size_t)i * i * sizeof(float));
        
        if (!a_cont || !b_cont || !c_cont) {
            printf("  allocation failed\n");
            free(a_cont); free(b_cont); free(c_cont);
            continue;
        }
        memset(a_cont, 0, (size_t)i * i * sizeof(float));
        memset(b_cont, 0, (size_t)i * i * sizeof(float));
        memset(c_cont, 0, (size_t)i * i * sizeof(float));

        int iters = 10;
        if (i <= 256) iters = 50;
        
        double w1 = bench_now();

        for (int iter = 0; iter < iters; iter++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a_cont, i, b_cont, i, 1.0f, c_cont, i);
        }

        dt = (float)((bench_now() - w1) / iters);

        printf(" %5.2f GFlops in %5.2fs\n", (float)2 * m * n * k / 1000000000 / dt, dt);
        
        free(a_cont);
        free(b_cont);
        free(c_cont);
    }
    
    // Free allocated memory
    free(a);
    free(b);
    free(c);
}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	cblas_init(CBLAS_DEFAULT_THREADS);
    cblas_print_configuration();
	
    test_gemm();

    cblas_print_stats();
    cblas_print_kernels();
    cblas_shutdown();

	return 0;
}
