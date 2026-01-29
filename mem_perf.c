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

#define MAX_SIZE 8192 * 8192 * 4

float x[MAX_SIZE];

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
void test_mem_read()
{
    struct timer t1, t2;
    float dt, v CBLAS_UNUSED = 0.0f;
    float* px = x;

    printf("Testing performance of memory reads\n\n");

    timer_get_time(&t1);
        for (int i = 0; i < MAX_SIZE; i++)
        {
            v = *px++;
        }
    timer_get_time(&t2);

    dt = timer_get_delta(&t1, &t2);

    int buf_size = 4 * MAX_SIZE / (1024 * 1024);
    printf("read %dMB at %5.2f MB/s\n", buf_size, (float)buf_size / dt);
}

//------------------------------------------------------
//
//------------------------------------------------------
void test_mem_write()
{
    struct timer t1, t2;
    float dt, v CBLAS_UNUSED = 0.0f;
    float* px CBLAS_UNUSED = x;

    printf("Testing performance of memory writes\n\n");

    timer_get_time(&t1);
        for (int i = 0; i < MAX_SIZE; i++)
        {
            x[i] = 0.0f;
        }
    timer_get_time(&t2);

    dt = timer_get_delta(&t1, &t2);

    int buf_size = 4 * MAX_SIZE / (1024 * 1024);
    printf("wrote %dMB at %5.2f MB/s\n", buf_size, (float)buf_size / dt);
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

    test_mem_read();
    test_mem_write();

	return 0;
}