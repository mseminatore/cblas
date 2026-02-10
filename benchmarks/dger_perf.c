//------------------------------------------------------
//
// Performance test for double-precision dger
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 2048

double x[MAX_SIZE], y[MAX_SIZE];
double a[MAX_SIZE * MAX_SIZE];

void test_dger(void)
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_dger() (double-precision)\n\n");

    CBLAS_INDEX m,n;

    for (int i = 64; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        // Initialize data
        for (CBLAS_INDEX j = 0; j < m; j++) {
            x[j] = 1.0;
            y[j] = 1.0;
        }
        for (CBLAS_INDEX j = 0; j < m * n; j++) {
            a[j] = 0.0;
        }

        cbu_timer_get_time(&t1);

        cblas_dger(CblasRowMajor, m, n, 1.0, x, 1, y, 1, a, m);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf("%4d x %4d: %5.2f GFlops in %5.2fs\n", i, i, (double)2 * m * n / 1000000000 / dt, dt);
    }
}

int main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	cblas_init(CBLAS_DEFAULT_THREADS);
	cblas_print_configuration();
	
    test_dger();

	cblas_shutdown();

	return 0;
}
