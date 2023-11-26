//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
//#include "test.h"
#include "cblas.h"


//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	cblas_init();

    printf( "%s\n", cblas_get_config());
    printf("      CPU uArch: %s\n", cblas_get_corename());
    printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
	
    struct timespec t1, t2;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);

    float x[1024], y[1024], a[1024*1024];

    cblas_sger(CblasRowMajor, 1024, 1024, 1.0f, x, 1, y, 1, a, 1024);

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t2);

    int seconds = (int)(t2.tv_sec - t1.tv_sec);
    long long ns = t2.tv_nsec - t1.tv_nsec;
    float dt = (float)seconds + (float)ns/(1000000000);

    printf("%f GLOPS in %fs\n", (float)2 * 0.001048f / dt, dt);

	return 0;
}