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

	cblas_scopy(ARRAY_SIZE(sbig_ones), sr1, 1, sbig_ones, 1);
	TEST(EQUAL_ARRAY(sbig_ones, sr1));

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
// test single and double precision ?ger()
//------------------------------------------------------
static void test_ger()
{
	SUITE("cblas_sger");

	float sx[3] = {1,1,1};
	float sy[3] = {1,1,1};
	float sa[9] = {0,0,0,0,0,0,0,0,0};
	float sr[9] = {1,1,1,1,1,1,1,1,1};

	cblas_sger(CblasRowMajor, 3, 3, 1.0f, sx, 1, sy, 1, sa, 3);
	TEST(EQUAL_ARRAY(sa, sr));

	cblas_sscal(9, 3.0f, sr, 1);
	cblas_sger(CblasColMajor, 3, 3, 2.0f, sx, 1, sy, 1, sa, 3);
	TEST(EQUAL_ARRAY(sa, sr));

	SUITE("cblas_dger");

	double dx[3] = {1,1,1};
	double dy[3] = {1,1,1};
	double da[9] = {0,0,0,0,0,0,0,0,0};
	double dr[9] = {1,1,1,1,1,1,1,1,1};

	cblas_dger(CblasRowMajor, 3, 3, 1.0, dx, 1, dy, 1, da, 3);
	TEST(EQUAL_ARRAY(da, dr));

	cblas_dscal(9, 3.0f, dr, 1);
	cblas_dger(CblasColMajor, 3, 3, 2.0, dx, 1, dy, 1, da, 3);
	TEST(EQUAL_ARRAY(da, dr));
}

//------------------------------------------------------
// test single and double precision ?gemv()
//------------------------------------------------------
static void test_gemv()
{
	SUITE("cblas_sgemv");

	float smtx[9] = {1,0,0,0,1,0,0,0,1};
	float sx[3] = {2,2,2};
	float sy[3] = {0,0,0};
	float sz[3] = {4, 4, 4};

	{
		float sy[3] = {0};
		cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0f, smtx, 1, sx, 1, 1.0f, sy, 1);
		TEST(EQUAL_ARRAY(sx, sy));
	}

	{
		float sy[3] = {0};
		cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 2.0f, smtx, 1, sx, 1, 1.0f, sy, 1);
		TEST(EQUAL_ARRAY(sz, sy));
	}

	{
		float sy[3] = {0};
		cblas_sgemv(CblasColMajor, CblasTrans, 3, 3, 1.0f, smtx, 1, sx, 1, 1.0f, sy, 1);
		TEST(EQUAL_ARRAY(sx, sy));
	}

	SUITE("cblas_dgemv");

	double dmtx[9] = {1,0,0,0,1,0,0,0,1};
	double dx[3] = {2,2,2};
	double dy[3] = {0,0,0};
	double dz[3] = {4,4,4};

	{
		double dy[3] = {0};
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, dmtx, 1, dx, 1, 1.0, dy, 1);
		TEST(EQUAL_ARRAY(dx, dy));
	}

	{
		double dy[3] = {0};
		cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 2.0, dmtx, 1, dx, 1, 1.0, dy, 1);
		TEST(EQUAL_ARRAY(dz, dy));
	}

	{
		double dy[3] = {0};
		cblas_dgemv(CblasColMajor, CblasTrans, 3, 3, 1.0, dmtx, 1, dx, 1, 1.0, dy, 1);
		TEST(EQUAL_ARRAY(dx, dy));
	}
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
static void test_gemm()
{
	SUITE("cblas_sgemm");

	float samtx[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	float sbmtx[] = {1,0,0,0,1,0,0,0,1};
	float scmtx[9] = {0};

print_sarray(9, samtx);
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 3, 3, 3, 1.0f, samtx, 3, sbmtx, 3, 1.0f, scmtx, 3);
print_sarray(9, scmtx);

	TEST(EQUAL_ARRAY(samtx, scmtx));

	SUITE("cblas_dgemm");

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

	test_gemm();
}

//------------------------------------------------------
//
//------------------------------------------------------
int test_main(int argc, char *argv[])
{
	cblas_init();

	printf( "%s\n", cblas_get_config());
	printf("      CPU uArch: %s\n", cblas_get_corename());
	printf("  Cores/Threads: %d/%d\n", cblas_get_num_procs(), cblas_get_num_threads());
	
	test_level1();
	test_level2();
	test_level3();

	return 0;
}
