//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "test.h"
#include "cblas.h"

#define size 1000000
#define checks 500
#define epsilon (1e-5 + 1e-8)

typedef float dtype;

dtype x[size];
dtype src[size][2];
dtype matrix[] = {0.0, 1.0, 1.0, 0.0};
dtype expected[size][2];
int mismatches[checks];
dtype result[size][2];

//------------------------------------------------------
//
//------------------------------------------------------
static int isclose()
{
	int count = 0;

	for (int i = 0; i < size; i++)
	{
		dtype delta = (dtype)fabs(result[i][0] - expected[i][0]);
		if (delta > epsilon)
			count++;

		delta = (dtype)fabs(result[i][1] - expected[i][1]);
		if (delta > epsilon)
			count++;
	}

	return count;
}

//
void clear_result()
{
	for (int i = 0; i < size; i++)
	{
		result[i][0] = 0.0;
		result[i][1] = 0.0;
	}
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_win_threads()
{
	for (int i = 0; i < size; i++)
	{
		x[i] = (dtype)i;
		src[i][0] = x[i];
		src[i][1] = (dtype)-10.0 * x[i];
		expected[i][0] = (dtype)-10.0 * x[i];
		expected[i][1] = x[i];
	}

	for (int i = 0; i < checks; i++)
	{
		clear_result();

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, size, 2, 2, 1.0f, (dtype*)src, 2, (dtype*)matrix, 2, 1.0f, (dtype*)result, 2);
		mismatches[i] = isclose();
		if (mismatches[i] > 0)
			printf("%d mismatching elements in multiplication %d", mismatches[i], i);
	}

	puts("done!");
}

//------------------------------------------------------
//
//------------------------------------------------------
int test_main(int argc, char *argv[])
{
//	cblas_set_num_threads(1);

	cblas_init(CBLAS_DEFAULT_THREADS);
	cblas_print_configuration();
	
	test_win_threads();

	return 0;
}