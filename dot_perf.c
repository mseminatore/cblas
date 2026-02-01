//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 65536*64

float x[MAX_SIZE], y[MAX_SIZE];

//------------------------------------------------------
//
//------------------------------------------------------
void test_dot(void)
{
    struct cblas_timer t1, t2;
    float dt;
    
    // Initialize arrays with non-zero values to avoid zero-page optimizations
    for (CBLAS_INDEX i = 0; i < MAX_SIZE; i++) {
        x[i] = (float)(i % 100) / 100.0f + 1.0f;
        y[i] = (float)(i % 100) / 100.0f + 1.0f;
    }

    printf("Testing performance of cblas_sdot()\n");
    printf("Note: Dot product is memory bandwidth limited (2 FLOPs per 8 bytes read)\n\n");
    printf("%10s %10s %12s %12s\n", "Size", "GFlops", "GB/s", "Time(s)");
    printf("-------------------------------------------------------\n");

    CBLAS_INDEX n;

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        n = i;

        cbu_timer_get_time(&t1);

        cblas_sdot(n, x, 1, y, 1);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);
        
        float gflops = (float)(2.0 * n) / dt / 1e9;
        float gbytes_per_sec = (float)(2.0 * n * sizeof(float)) / dt / 1e9;

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
	cblas_print_kernels();
	printf("\n");
	
    test_dot();

	cblas_print_stats();
	cblas_shutdown();

	return 0;
}
