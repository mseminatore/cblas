//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include "test.h"
#include "cblas.h"

#if defined(_WIN32) | defined(_WIN64)
#	include <Windows.h>
#endif

//
int test_number = 0;
int test_failures = 0;
int test_suites = 0;
int test_modules = 0;

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
static void print_sarray(int n, float *x)
{
	putchar('{');
	
	for (int i = 0; i < n; i++)
	{
		printf("%f, ", x[i]);
	}

	puts("}");
}

//------------------------------------------------------
//
//------------------------------------------------------
static void print_darray(int n, double* x)
{
	putchar('{');

	for (int i = 0; i < n; i++)
	{
		printf("%g, ", x[i]);
	}

	puts("}");
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_swap()
{
	SUITE("cblas_sswap");

	// swap two vectors
	cblas_sswap(ARRAY_SIZE(sa), sa, 1, sc, 1);
	TEST(EQUAL_ARRAY(sa, sd));
	TEST(EQUAL_ARRAY(sb, sc));

	// swap them back
	cblas_sswap(ARRAY_SIZE(sa), sa, 1, sc, 1);
	TEST(EQUAL_ARRAY(sa, sb));
	TEST(EQUAL_ARRAY(sc, sd));

	SUITE("cblas_dswap");

	// swap two vectors
	cblas_dswap(ARRAY_SIZE(da), da, 1, dc, 1);
	TEST(EQUAL_ARRAY(da, dd));
	TEST(EQUAL_ARRAY(db, dc));

	// swap them back
	cblas_dswap(ARRAY_SIZE(da), da, 1, dc, 1);
	TEST(EQUAL_ARRAY(da, db));
	TEST(EQUAL_ARRAY(dc, dd));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_dot()
{
	SUITE("cblas_sdot");

	float sr = cblas_sdot(ARRAY_SIZE(szeros), szeros, 1, sones, 1);
	TEST(0.0f == sr);

	sr = cblas_sdot(ARRAY_SIZE(szeros), szeros, 1, szeros, 1);
	TEST(0.0f == sr);

	sr = cblas_sdot(ARRAY_SIZE(sones), sones, 1, sones, 1);
	TEST(ARRAY_SIZE(sones) == sr);

	SUITE("cblas_ddot");

	double dr = cblas_ddot(ARRAY_SIZE(dzeros), dzeros, 1, dones, 1);
	TEST(0.0 == dr);

	dr = cblas_ddot(ARRAY_SIZE(dzeros), dzeros, 1, dzeros, 1);
	TEST(0.0 == dr);

	dr = cblas_ddot(ARRAY_SIZE(dones), dones, 1, dones, 1);
	TEST(ARRAY_SIZE(dones) == dr);
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_copy()
{
	SUITE("cblas_scopy");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(sones), sr, 1, sones, 1);
	TEST(EQUAL_ARRAY(sones, sr));

	float sr1[ARRAY_SIZE(sbig_ones)];

	//time_t time_start = time(NULL);

	cblas_scopy(ARRAY_SIZE(sbig_ones), sr1, 1, sbig_ones, 1);
	TEST(EQUAL_ARRAY(sbig_ones, sr1));

//	time_t time_end = time(NULL);
//	double diff_t = (double)(time_end - time_start);
//	double per_step = 1000.0 * diff_t / (rows * epoch);
//	printf("\ncblas_scopy time: %f seconds\n", diff_t);

	SUITE("cblas_dcopy");

	double dr[ARRAY_SIZE(dones)];

	cblas_dcopy(ARRAY_SIZE(dones), dr, 1, dones, 1);
	TEST(EQUAL_ARRAY(dones, dr));

	double dr1[ARRAY_SIZE(dbig_ones)];

	cblas_dcopy(ARRAY_SIZE(dbig_ones), dr1, 1, dbig_ones, 1);
	TEST(EQUAL_ARRAY(dbig_ones, dr1));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_axpy()
{
	SUITE("cblas_saxpy");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(szeros), sr, 1, szeros, 1);
	cblas_saxpy(ARRAY_SIZE(sones), 1.0f, sones, 1, sr, 1);
	TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_daxpy");

	double dr[ARRAY_SIZE(dones)];

	cblas_dcopy(ARRAY_SIZE(dzeros), dr, 1, dzeros, 1);
	cblas_daxpy(ARRAY_SIZE(dones), 1.0, dones, 1, dr, 1);
	TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_axpby()
{
	SUITE("cblas_saxpby");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(szeros), sr, 1, szeros, 1);
	cblas_saxpby(ARRAY_SIZE(sones), 1.0f, sones, 1, 1.0f, sr, 1);
	TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_daxpby");

	double dr[ARRAY_SIZE(dones)];

	cblas_dcopy(ARRAY_SIZE(dzeros), dr, 1, dzeros, 1);
	cblas_daxpby(ARRAY_SIZE(dones), 1.0, dones, 1, 1.0, dr, 1);
	TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_scal()
{
	SUITE("cblas_sscal");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(sones), sr, 1, sones, 1);
	cblas_sscal(ARRAY_SIZE(sr), 1.0, sr, 1);
	TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_dscal");

	double dr[ARRAY_SIZE(dones)];

	cblas_dcopy(ARRAY_SIZE(dones), dr, 1, dones, 1);
	cblas_dscal(ARRAY_SIZE(dr), 1.0, dr, 1);
	TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_rot()
{
	SUITE("cblas_srot");

	SUITE("cblas_drot");
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_asum()
{
	SUITE("cblas_sasum");

	// sum of zeros is zero
	float sr = cblas_sasum(ARRAY_SIZE(szeros), szeros, 1);
	TEST(sr == 0.0f);

	// sum of ones is count(ones)
	sr = cblas_sasum(ARRAY_SIZE(sones), sones, 1);
	TEST(sr == ARRAY_SIZE(sones) * 1.0f);

	sr = cblas_sasum(ARRAY_SIZE(sa), sa, 1);
	TEST(sr == 45.0f);

	SUITE("cblas_dasum");

	// sum of zeros is zero
	double dr = cblas_dasum(ARRAY_SIZE(dzeros), dzeros, 1);
	TEST(dr == 0.0);

	// sum of ones is count(ones)
	dr = cblas_dasum(ARRAY_SIZE(dones), dones, 1);
	TEST(dr == ARRAY_SIZE(dones) * 1.0f);

	dr = cblas_dasum(ARRAY_SIZE(da), da, 1);
	TEST(sr == 45.0);
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_nrm2()
{
	SUITE("cblas_snrm2");

	float sr = cblas_snrm2(ARRAY_SIZE(szeros), szeros, 1);
	TEST(sr == 0.0f);

	sr = cblas_snrm2(ARRAY_SIZE(sones), sones, 1);
	TEST(sr * sr == 10.0f);

	sr = cblas_snrm2(ARRAY_SIZE(sa), sa, 1);
	TEST(sr * sr == 285.0f);

	SUITE("cblas_dnrm2");

	double dr = cblas_dnrm2(ARRAY_SIZE(dzeros), dzeros, 1);
	TEST(dr == 0.0);

	dr = cblas_dnrm2(ARRAY_SIZE(dones), dones, 1);
	TEST(EQUAL_EPSILON(dr * dr, 10.0));

	dr = cblas_dnrm2(ARRAY_SIZE(da), da, 1);
	TEST(EQUAL_EPSILON(dr * dr, 285.0));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_ger()
{
	SUITE("cblas_sger");

	SUITE("cblas_dger");
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_gemv()
{
	SUITE("cblas_sgemv");

	SUITE("cblas_dgemv");
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_rotg()
{
	SUITE("cblas_srotg");

	SUITE("cblas_drotg");

}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_level1()
{
	MODULE("BLAS Level1");

	test_swap();
	test_dot();
	test_copy();
	test_axpy();
	test_scal();
	test_axpby();
	test_rot();
	test_asum();
	test_nrm2();
	test_rotg();
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_level2()
{
	MODULE("BLAS Level2");

	test_ger();
	test_gemv();
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_level3()
{
	MODULE("BLAS Level3");

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

	test_level1();
	test_level2();
	test_level3();

	END_TESTS();
}
