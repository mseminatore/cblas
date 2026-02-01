//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 8192

float x[MAX_SIZE], y[MAX_SIZE];
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
    }
    for (CBLAS_INDEX i = 0; i < MAX_SIZE * MAX_SIZE; i++) {
        a[i] = (float)(i % 100) / 100.0f + 1.0f;
    }

    printf("Testing performance of cblas_sgemv()\n");
    printf("Note: GEMV performs matrix-vector multiply: y = alpha * A * x + beta * y\n\n");
    printf("%10s %10s %12s %12s\n", "Size", "GFlops", "GB/s", "Time(s)");
    printf("-------------------------------------------------------\n");

    CBLAS_INDEX m, n;

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        cbu_timer_get_time(&t1);

        cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0f, a, n, x, 1, 1.0f, y, 1);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);
        
        // GEMV does 2*m*n FLOPs (multiply-add for each matrix element)
        float gflops = (float)(2.0 * m * n) / dt / 1e9;
        // Memory: read m*n from A, n from x, m from y, write m to y
        float gbytes_per_sec = (float)((m*n + n + 2*m) * sizeof(float)) / dt / 1e9;

        printf("%10d %10.2f %12.2f %12.6f\n", i, gflops, gbytes_per_sec, dt);
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

	cblas_shutdown();

	return 0;
}
