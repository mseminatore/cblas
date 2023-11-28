//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
//#include "test.h"
#include "cblas.h"

#define MAX_SIZE 1024

float a[MAX_SIZE * MAX_SIZE], b[MAX_SIZE * MAX_SIZE], c[MAX_SIZE * MAX_SIZE];

//------------------------------------------------------
//
//------------------------------------------------------
void test_gemm()
{
   struct timespec t1, t2;

    CBLAS_INDEX m, n, k;

    for (int i = 2; i <= MAX_SIZE; i <<= 1)
    {
#ifndef WIN32
        m = n = k = i;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, m, b, k, 1.0f, c, k);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        int seconds = (int)(t2.tv_sec - t1.tv_sec);
        long long ns = t2.tv_nsec - t1.tv_nsec;
        float dt = (float)seconds + (float)ns/(1000000000);

        printf("%d: %f GFlops in %fs\n", i, (float)2 * m * n / 1000000000 / dt, dt);
#endif
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_ger()
{
    struct timespec t1, t2;

    CBLAS_INDEX m = MAX_SIZE, n = MAX_SIZE;
    float x[MAX_SIZE], y[MAX_SIZE], a[MAX_SIZE * MAX_SIZE];

    for (int i = 2; i <= MAX_SIZE; i <<= 1)
    {
#ifndef WIN32
        m = n = i;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, m);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

        int seconds = (int)(t2.tv_sec - t1.tv_sec);
        long long ns = t2.tv_nsec - t1.tv_nsec;
        float dt = (float)seconds + (float)ns/(1000000000);

        printf("%d: %f GFlops in %fs\n", i, (float)2 * m * n / 1000000000 / dt, dt);
#endif
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	cblas_init();

    printf( "%s\n", cblas_get_config());
    printf("    CPU uArch: %s\n", cblas_get_corename());
    printf("Cores/Threads: %d/%d\n\n", cblas_get_num_procs(), cblas_get_num_threads());
	
    test_gemm();

	return 0;
}