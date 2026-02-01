//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

#define MAX_SIZE 8192

//------------------------------------------------------
//
//------------------------------------------------------
void test_gemm(void)
{
    struct cblas_timer t1, t2;
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

        cbu_timer_get_time(&t1);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, MAX_SIZE, b, MAX_SIZE, 1.0f, c, MAX_SIZE);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf(" %5.2f GFlops in %5.2fs\n", (float)2 * m * n * k / 1000000000 / dt, dt);
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
	cblas_shutdown();

	return 0;
}
