//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "test.h"
#include "cblas.h"

static float szeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static float sones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static float sa[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sb[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static float sd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

static float sbig_ones[1024] = { 1.0f };
static float sbig_zeroes[1024] = { 0.0f };

static double dbig_ones[1024] = { 1.0 };
static double dbig_zeroes[1024] = { 0.0 };

static double dzeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static double dones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static double da[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double db[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double dc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static double dd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

//------------------------------------------------------
//
//------------------------------------------------------
static void test_stress()
{
	MODULE("BLAS Level 1 stress test");

	SUITE("cblas_scopy stress");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(sones), sr, 1, sones, 1);
	TEST(EQUAL_ARRAY(sones, sr));

	float sr1[ARRAY_SIZE(sbig_ones)];

	time_t time_start = time(NULL);

	for (int i = 0; i < 165000; i++)
	{
		cblas_scopy(ARRAY_SIZE(sbig_ones), sr1, 1, sbig_ones, 1);
	}
	TEST(EQUAL_ARRAY(sbig_ones, sr1));

	time_t time_end = time(NULL);
	double diff_t = (double)(time_end - time_start);
//	double per_step = 1000.0 * diff_t / (rows * epoch);
	printf("\ncblas_scopy time: %f seconds\n", diff_t);
}

//------------------------------------------------------
//
//------------------------------------------------------
int test_main(int argc, char *argv[])
{
//	cblas_set_num_threads(1);

	cblas_init();
	
	test_stress();
}