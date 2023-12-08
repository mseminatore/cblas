//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#ifdef WIN32
#   include <Windows.h>
#endif

#define MAX_SIZE 1024

float x[MAX_SIZE], y[MAX_SIZE];
float a[MAX_SIZE * MAX_SIZE];

struct timer
{
#ifdef WIN32
    LARGE_INTEGER t;
#else
    struct timespec t;
#endif

};

//------------------------------------------------------
//
//------------------------------------------------------
void timer_get_time(struct timer* t)
{
#ifdef WIN32
    QueryPerformanceCounter(&t->t);
#else
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t->t);
#endif

}

//------------------------------------------------------
//
//------------------------------------------------------
float timer_get_delta(struct timer *t1, struct timer *t2)
{
    float dt;

#ifdef WIN32
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);

    dt = (t2->t.QuadPart - t1->t.QuadPart) / (float)freq.QuadPart;

#else
    int seconds = (int)(t2->t.tv_sec - t1->t.tv_sec);
    long long ns = t2->t.tv_nsec - t1->t.tv_nsec;
    dt = (float)seconds + (float)ns / (1000000000);
#endif

    return dt;
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_ger()
{
    struct timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sger()\n\n");

    CBLAS_INDEX m,n;

    for (int i = 2; i <= MAX_SIZE; i <<= 1)
    {
        m = n = i;

        timer_get_time(&t1);

        cblas_sger(CblasRowMajor, m, n, 1.0f, x, 1, y, 1, a, m);

        timer_get_time(&t2);

        dt = timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * m * n / 1000000000 / dt, dt);
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
	
    test_ger();

	return 0;
}