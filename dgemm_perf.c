//------------------------------------------------------
//
// Performance test for double-precision dgemm
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "cblas.h"

#define MAX_SIZE 1024

void test_dgemm(void)
{
    struct cblas_timer t1, t2;
    CBLAS_INDEX m, n, k;
    float dt;
    
    // Allocate matrices on heap to avoid stack overflow
    double *a = (double*)malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
    double *b = (double*)malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
    double *c = (double*)malloc(MAX_SIZE * MAX_SIZE * sizeof(double));
    
    if (!a || !b || !c) {
        fprintf(stderr, "Failed to allocate matrices\n");
        free(a); free(b); free(c);
        return;
    }
    
    // Initialize matrices
    memset(a, 0, MAX_SIZE * MAX_SIZE * sizeof(double));
    memset(b, 0, MAX_SIZE * MAX_SIZE * sizeof(double));
    memset(c, 0, MAX_SIZE * MAX_SIZE * sizeof(double));

    printf("Testing performance of cblas_dgemm() (double-precision)\n\n");

    for (int i = 64; i <= MAX_SIZE; i <<= 1)
    {
        m = n = k = i;

        // Initialize data
        for (CBLAS_INDEX j = 0; j < m * k; j++) {
            a[j] = 1.0;
        }
        for (CBLAS_INDEX j = 0; j < k * n; j++) {
            b[j] = 1.0;
        }
        for (CBLAS_INDEX j = 0; j < m * n; j++) {
            c[j] = 0.0;
        }

        cbu_timer_get_time(&t1);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, MAX_SIZE, b, MAX_SIZE, 1.0, c, MAX_SIZE);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf("%4d x %4d: %5.2f GFlops in %5.2fs\n", i, i, (double)2 * m * n * k / 1000000000 / dt, dt);
    }
    
    // Free allocated memory
    free(a);
    free(b);
    free(c);
}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	cblas_init(CBLAS_DEFAULT_THREADS);
	cblas_print_configuration();
	
    test_dgemm();

	cblas_shutdown();

	return 0;
}
