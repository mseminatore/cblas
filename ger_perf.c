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
void test_ger()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sger()\n\n");

    CBLAS_INDEX m,n;

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        cbu_timer_get_time(&t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, m);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * m * n / 1000000000 / dt, dt);
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	cblas_init(CBLAS_DEFAULT_THREADS);
	cblas_print_configuration();
	
    test_ger();

	return 0;
}