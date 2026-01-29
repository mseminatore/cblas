//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "cblas.h"

#define MAX_SIZE 65536*64

float x[MAX_SIZE], y[MAX_SIZE];

//------------------------------------------------------
//
//------------------------------------------------------
void test_dot()
{
    struct cblas_timer t1, t2;
    float dt;

    printf("Testing performance of cblas_sdot()\n\n");

    CBLAS_INDEX n;

    for (int i = 4; i <= MAX_SIZE; i <<= 1)
    {
        n = i;

        cbu_timer_get_time(&t1);

        cblas_sdot(n, x, 1, y, 1);

        cbu_timer_get_time(&t2);

        dt = cbu_timer_get_delta(&t1, &t2);

        printf("%4d: %5.2f GFlops in %5.2fs\n", i, (float)2 * n / 1000000000 / dt, dt);
    }
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
	
    test_dot();

	return 0;
}