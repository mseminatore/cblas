//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "cblas.h"

#define MAX_SIZE 65536*64

float x[MAX_SIZE], y[MAX_SIZE];

//------------------------------------------------------
// Test nrm2 performance
//------------------------------------------------------
void test_nrm2()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_snrm2()\n\n");

    CBLAS_INDEX n;

    for (int i = 16; i <= MAX_SIZE; i <<= 1)
    {
        n = i;

        // Initialize test data
        for (CBLAS_INDEX j = 0; j < n; j++)
            x[j] = 1.0f;

        cbu_timer_get_time(&t1);

        float result = cblas_snrm2(n, x, 1);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        // nrm2 performs 2*n-1 operations (n muls + n-1 adds) + 1 sqrt
        // Note: sqrt is more expensive than mul/add, so GFlops may not reflect true throughput
        printf("%8d: %5.2f GFlops in %8.6fs (result=%f)\n", i, (float)(2 * n) / 1000000000 / dt, dt, result);
    }
    printf("\n");
}

//------------------------------------------------------
// Test asum performance
//------------------------------------------------------
void test_asum()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sasum()\n\n");

    CBLAS_INDEX n;

    for (int i = 16; i <= MAX_SIZE; i <<= 1)
    {
        n = i;

        // Initialize test data
        for (CBLAS_INDEX j = 0; j < n; j++)
            x[j] = (j % 2 == 0) ? 1.0f : -1.0f;

        cbu_timer_get_time(&t1);

        float result = cblas_sasum(n, x, 1);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        // asum performs 2*n-1 operations (n abs + n-1 adds)
        // Note: abs implementation (bit masking vs branching) affects actual throughput
        printf("%8d: %5.2f GFlops in %8.6fs (result=%f)\n", i, (float)(2 * n) / 1000000000 / dt, dt, result);
    }
    printf("\n");
}

//------------------------------------------------------
// Test rot performance
//------------------------------------------------------
void test_rot()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_srot()\n\n");

    CBLAS_INDEX n;

    float c = cosf(M_PI / 4.0f);
    float s = sinf(M_PI / 4.0f);

    for (int i = 16; i <= MAX_SIZE; i <<= 1)
    {
        n = i;

        // Initialize test data
        for (CBLAS_INDEX j = 0; j < n; j++)
        {
            x[j] = 1.0f;
            y[j] = 1.0f;
        }

        cbu_timer_get_time(&t1);

        cblas_srot(n, x, 1, y, 1, c, s);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        // rot performs 6*n operations (4 muls + 2 add/subs per element)
        printf("%8d: %5.2f GFlops in %8.6fs (x[0]=%f, y[0]=%f)\n", i, (float)(6 * n) / 1000000000 / dt, dt, x[0], y[0]);
    }
    printf("\n");
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
	
    test_nrm2();
    test_asum();
    test_rot();

	cblas_shutdown();

	return 0;
}
