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
float a[MAX_SIZE * MAX_SIZE], b[MAX_SIZE * MAX_SIZE];

//------------------------------------------------------
//
//------------------------------------------------------
// void test_copy()
// {
//     struct cblas_timer t1, t2;
//     float dt;

//     printf("Testing performance of cblas_scopy()\n\n");

//     CBLAS_INDEX n = MAX_SIZE * MAX_SIZE;

//     cbu_timer_get_time(&t1);
//         cblas_scopy(n, a, 1, b, 1);
//     cbu_timer_get_time(&t2);

//     dt = cbu_timer_get_delta(&t1, &t2);

//     printf("copied %dMB at %5.2f MB/s\n", 1, (float)1 / dt);
// }

//------------------------------------------------------
//
//------------------------------------------------------
void test_ger(void)
{
    struct cblas_timer t1, t2;
    float dt;
    
    // Initialize arrays with non-zero values
    for (CBLAS_INDEX i = 0; i < MAX_SIZE; i++) {
        x[i] = (float)(i % 100) / 100.0f + 1.0f;
        y[i] = (float)(i % 100) / 100.0f + 1.0f;
    }
    for (CBLAS_INDEX i = 0; i < MAX_SIZE * MAX_SIZE; i++) {
        a[i] = 0.0f;
    }

    printf("Testing performance of cblas_sger()\n");
    printf("Note: GER performs outer product (rank-1 update): A += x * y^T\n\n");
    printf("%10s %10s %12s %12s\n", "Size", "GFlops", "GB/s", "Time(s)");
    printf("-------------------------------------------------------\n");

    CBLAS_INDEX m,n;

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        cbu_timer_get_time(&t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, n);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);
        
        // GER does m*n FMA operations (multiply-add)
        float gflops = (float)(2.0 * m * n) / dt / 1e9;
        // Memory: read m elements from x, n from y, m*n from A, write m*n to A
        float gbytes_per_sec = (float)((m + n + 2*m*n) * sizeof(float)) / dt / 1e9;

        printf("%10d %10.2f %12.2f %12.6f\n", i, gflops, gbytes_per_sec, dt);
    }
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
	
    test_ger();

	cblas_shutdown();

	return 0;
}
