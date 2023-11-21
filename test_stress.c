//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "test.h"
#include "cblas.h"

#if defined(_WIN32) | defined(_WIN64)
#	include <Windows.h>
#endif

//
int test_number		= 0;
int test_failures 	= 0;
int test_suites 	= 0;
int test_modules 	= 0;

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
// enable VT100 support in pre Win11 console window
//------------------------------------------------------
int setupConsole()
{
#if defined(_WIN32) || defined(_WIN64)
	// Set output mode to handle virtual terminal sequences
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE)
	{
		return GetLastError();
	}

	DWORD dwMode = 0;
	if (!GetConsoleMode(hOut, &dwMode))
	{
		return GetLastError();
	}

	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(hOut, dwMode))
	{
		return GetLastError();
	}
#endif

	return 0;
}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	setupConsole();

	cblas_init();
	
	BEGIN_TESTS();

	test_stress();

	END_TESTS();
}