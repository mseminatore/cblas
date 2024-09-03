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
float a[MAX_SIZE * MAX_SIZE], b[MAX_SIZE * MAX_SIZE], c[MAX_SIZE * MAX_SIZE];

//------------------------------------------------------
//
//------------------------------------------------------
void test_gemm()
{
    struct cblas_timer t1, t2;
    CBLAS_INDEX m, n, k;
    float dt;

    printf("Testing performance of cblas_sgemm()\n\n");

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        m = n = k = i;

        cblas_timer_get_time(&t1);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, m, b, k, 1.0f, c, k);

        cblas_timer_get_time(&t2);

        dt = cblas_timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * m * n * k / 1000000000 / dt, dt);
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_ger()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sger()\n\n");

    CBLAS_INDEX m = MAX_SIZE, n = MAX_SIZE;

    for (int i = 2; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        cblas_timer_get_time(&t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, m);

        cblas_timer_get_time(&t2);

        dt = cblas_timer_get_delta(&t1, &t2);

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
	
    test_gemm();

	return 0;
}