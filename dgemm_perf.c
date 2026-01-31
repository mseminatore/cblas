//------------------------------------------------------
//
// Performance test for double-precision dgemm
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include "cblas.h"

#define MAX_SIZE 1024

double a[MAX_SIZE * MAX_SIZE], b[MAX_SIZE * MAX_SIZE], c[MAX_SIZE * MAX_SIZE];

void test_dgemm(void)
{
    struct cblas_timer t1, t2;
    CBLAS_INDEX m, n, k;
    float dt;

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

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 1.0, c, n);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf("%4d x %4d: %5.2f GFlops in %5.2fs\n", i, i, (double)2 * m * n * k / 1000000000 / dt, dt);
    }
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
