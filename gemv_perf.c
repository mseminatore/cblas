//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 8192
#define WARMUP_ITERS 3
#define BENCH_ITERS 10

float x[MAX_SIZE], y[MAX_SIZE], y_copy[MAX_SIZE];
float a[MAX_SIZE * MAX_SIZE];

//------------------------------------------------------
// Test GEMV performance
//------------------------------------------------------
void test_gemv(void)
{
    struct cblas_timer t1, t2;
    float dt;
    
    // Initialize arrays with non-zero values to avoid compiler optimizations
    for (CBLAS_INDEX i = 0; i < MAX_SIZE; i++) {
        x[i] = (float)(i % 100) / 100.0f + 1.0f;
        y[i] = (float)(i % 100) / 100.0f + 1.0f;
        y_copy[i] = y[i];
    }
    for (CBLAS_INDEX i = 0; i < MAX_SIZE * MAX_SIZE; i++) {
        a[i] = (float)(i % 100) / 100.0f + 1.0f;
    }

    printf("Testing performance of cblas_sgemv()\n");
    printf("Note: GEMV performs matrix-vector multiply: y = alpha * A * x + beta * y\n");
    printf("Warmup: %d iterations, Benchmark: %d iterations (best time reported)\n\n", WARMUP_ITERS, BENCH_ITERS);
    printf("%10s %10s %12s %12s %8s\n", "Size", "GFlops", "GB/s", "Time(s)", "MT");
    printf("--------------------------------------------------------------\n");

    CBLAS_INDEX m, n;

    for (int size = 4; size <= MAX_SIZE; size <<= 1)
    {
        m = n = size;

        // Warmup iterations - prime caches and thread pool
        for (int w = 0; w < WARMUP_ITERS; w++) {
            cblas_scopy(m, y_copy, 1, y, 1);  // Reset y
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, x, 1, 1.0f, y, 1);
        }

        // Benchmark iterations - take best time
        float best_time = 1e30f;
        for (int iter = 0; iter < BENCH_ITERS; iter++) {
            cblas_scopy(m, y_copy, 1, y, 1);  // Reset y
            
            cbu_timer_get_time(&t1);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, x, 1, 1.0f, y, 1);
            cbu_timer_get_time(&t2);

            dt = cbu_timer_get_delta(&t1, &t2);
            if (dt < best_time) best_time = dt;
        }
        
        // GEMV does 2*m*n FLOPs (multiply-add for each matrix element)
        float gflops = (float)(2.0f * m * n) / best_time / 1e9f;
        // Memory: read m*n from A, n from x, m from y, write m to y
        float gbytes_per_sec = (float)((m*n + n + 2*m) * sizeof(float)) / best_time / 1e9f;
        // Check if MT would be used
        const char* mt_flag = ((long long)m * n > CBLAS_MT_GEMV) ? "yes" : "no";

        printf("%10d %10.2f %12.2f %12.6f %8s\n", size, gflops, gbytes_per_sec, best_time, mt_flag);
    }
}

//------------------------------------------------------
// Main
//------------------------------------------------------
int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	cblas_init(CBLAS_DEFAULT_THREADS);
	cblas_print_configuration();
	
    test_gemv();

    cblas_print_stats();
	cblas_shutdown();

	return 0;
}
