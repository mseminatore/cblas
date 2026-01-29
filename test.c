//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_MSC_VER)
#	define _USE_MATH_DEFINES
#endif

#include <math.h>
#include "test.h"
#include "cblas.h"

static float szeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static float sones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static float sa[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sb[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static float sd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

#define BIG_ARRAY 65536
static float *sbig_ones;
static float *sbig_zeroes;

static double *dbig_ones;
static double *dbig_zeroes;

static double dzeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static double dones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static double da[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double db[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double dc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static double dd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

//------------------------------------------------------
//
//------------------------------------------------------
int equal_sarray_epsilon(float *a, float *b, int len)
{
	for (int i = 0; i < len; i++)
	{
		if (!EQUAL_EPSILON(a[i], b[i]))
			return 0;
	}

	return 1;
}

int equal_darray_epsilon(double* a, double* b, int len)
{
	for (int i = 0; i < len; i++)
	{
		if (!EQUAL_EPSILON(a[i], b[i]))
			return 0;
	}

	return 1;
}

//------------------------------------------------------
//
//------------------------------------------------------
static float* svec_fill(CBLAS_INDEX size, float val)
{
	float* result = malloc(size * sizeof(float));
	if (!result)
		return result;

	for (CBLAS_INDEX i = 0; i < size; i++)
		result[i] = val;

	return result;

}

//------------------------------------------------------
//
//------------------------------------------------------
static double* dvec_fill(CBLAS_INDEX size, double val)
{
	double* result = malloc(size * sizeof(double));
	if (!result)
		return result;

	for (CBLAS_INDEX i = 0; i < size; i++)
		result[i] = val;

	return result;

}

//------------------------------------------------------
//
//------------------------------------------------------
static float* svec_zeroes(CBLAS_INDEX size)
{
	return svec_fill(size, 0.0f);
}

//------------------------------------------------------
//
//------------------------------------------------------
static float* svec_ones(CBLAS_INDEX size)
{
	return svec_fill(size, 1.0f);
}

//------------------------------------------------------
//
//------------------------------------------------------
static double* dvec_zeroes(CBLAS_INDEX size)
{
	return dvec_fill(size, 0.0);
}

//------------------------------------------------------
//
//------------------------------------------------------
static double* dvec_ones(CBLAS_INDEX size)
{
	return dvec_fill(size, 1.0);
}

//------------------------------------------------------
//
//------------------------------------------------------
CBLAS_UNUSED static void print_sarray(int n, float *x)
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
CBLAS_UNUSED static void print_darray(int n, double* x)
{
	putchar('{');

	for (int i = 0; i < n; i++)
	{
		printf("%g, ", x[i]);
	}

	puts("}");
}

//------------------------------------------------------
// test cases for s/dswap()
//------------------------------------------------------
static void test_swap(void)
{
	SUITE("cblas_sswap");

	// swap two vectors
	cblas_sswap(ARRAY_SIZE(sa), sa, 1, sc, 1);
	 COMMENT("Swap two vectors");
	TEST(EQUAL_ARRAY(sa, sd));
	TEST(EQUAL_ARRAY(sb, sc));

	// swap them back
	cblas_sswap(ARRAY_SIZE(sa), sa, 1, sc, 1);
	 COMMENT("Swap them back");
	TEST(EQUAL_ARRAY(sa, sb));
	TEST(EQUAL_ARRAY(sc, sd));

	// TODO - swap using incx/incy > 1
// print_sarray(ARRAY_SIZE(sa), sa);
// print_sarray(ARRAY_SIZE(sa), sc);
 	//cblas_sswap(ARRAY_SIZE(sa), sa, 2, sc, 2);
// print_sarray(ARRAY_SIZE(sa), sa);
// print_sarray(ARRAY_SIZE(sa), sc);

// 	cblas_sswap(ARRAY_SIZE(sa), sa, 2, sc, 2);
 	TEST(EQUAL_ARRAY(sa, sb));
// print_sarray(ARRAY_SIZE(sa), sa);
// print_sarray(ARRAY_SIZE(sa), sc);

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
// test cases for s/ddot()
//------------------------------------------------------
static void test_dot(void)
{
	SUITE("cblas_sdot");

		float sr = cblas_sdot(ARRAY_SIZE(szeros), szeros, 1, sones, 1);
		TEST(0.0f == sr);

		sr = cblas_sdot(ARRAY_SIZE(szeros), szeros, 1, szeros, 1);
		TEST(0.0f == sr);

		sr = cblas_sdot(ARRAY_SIZE(sones), sones, 1, sones, 1);
		TEST(ARRAY_SIZE(sones) == sr);

		sr = cblas_sdot(BIG_ARRAY, sbig_ones, 1, sbig_ones, 1);
		TEST(sr == BIG_ARRAY);

		sr = cblas_sdot(BIG_ARRAY, sbig_ones, 1, sbig_zeroes, 1);
		TEST(sr == 0.0f);

	SUITE("cblas_ddot");

		double dr = cblas_ddot(ARRAY_SIZE(dzeros), dzeros, 1, dones, 1);
		TEST(0.0 == dr);

		dr = cblas_ddot(ARRAY_SIZE(dzeros), dzeros, 1, dzeros, 1);
		TEST(0.0 == dr);

		dr = cblas_ddot(ARRAY_SIZE(dones), dones, 1, dones, 1);
		TEST(ARRAY_SIZE(dones) == dr);

		dr = cblas_ddot(BIG_ARRAY, dbig_ones, 1, dbig_ones, 1);
		TEST(dr == BIG_ARRAY);

		dr = cblas_ddot(BIG_ARRAY, dbig_ones, 1, dbig_zeroes, 1);
		TEST(dr == 0.0f);
}

//------------------------------------------------------
// test cases for s/dcopy()
//------------------------------------------------------
static void test_copy(void)
{
	SUITE("cblas_scopy");

		float sr[ARRAY_SIZE(sones)];

		cblas_scopy(ARRAY_SIZE(sones), sones, 1, sr, 1);
		TEST(EQUAL_ARRAY(sones, sr));

		float sr1[BIG_ARRAY];

		cblas_scopy(BIG_ARRAY, sbig_ones, 1, sr1, 1);
		TEST(EQUAL_ARRAY_SIZE(sbig_ones, sr1, BIG_ARRAY));

	SUITE("cblas_dcopy");

		double dr[ARRAY_SIZE(dones)];

		cblas_dcopy(ARRAY_SIZE(dones), dones, 1, dr, 1);
		TEST(EQUAL_ARRAY(dones, dr));

		double dr1[BIG_ARRAY];

		cblas_dcopy(BIG_ARRAY, dbig_ones, 1, dr1, 1);
		TEST(EQUAL_ARRAY_SIZE(dbig_ones, dr1, BIG_ARRAY));
}

//------------------------------------------------------
// test cases for s/daxpy()
//------------------------------------------------------
static void test_axpy(void)
{
	SUITE("cblas_saxpy");

		float sr[ARRAY_SIZE(sones)];

		cblas_scopy(ARRAY_SIZE(szeros), szeros, 1, sr, 1);
		cblas_saxpy(ARRAY_SIZE(sones), 1.0f, sones, 1, sr, 1);
		TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_daxpy");

		double dr[ARRAY_SIZE(dones)];

		cblas_dcopy(ARRAY_SIZE(dzeros), dzeros, 1, dr, 1);
		cblas_daxpy(ARRAY_SIZE(dones), 1.0, dones, 1, dr, 1);
		TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
// test cases for s/daxpby()
//------------------------------------------------------
static void test_axpby(void)
{
	SUITE("cblas_saxpby");

		float sr[ARRAY_SIZE(sones)];

		cblas_scopy(ARRAY_SIZE(szeros), szeros, 1, sr, 1);
		cblas_saxpby(ARRAY_SIZE(sones), 1.0f, sones, 1, 1.0f, sr, 1);
		TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_daxpby");

		double dr[ARRAY_SIZE(dones)];

		cblas_dcopy(ARRAY_SIZE(dzeros), dzeros, 1, dr, 1);
		cblas_daxpby(ARRAY_SIZE(dones), 1.0, dones, 1, 1.0, dr, 1);
		TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
// test cases for s/dscal()
//------------------------------------------------------
static void test_scal(void)
{
	SUITE("cblas_sscal");

		float sr[ARRAY_SIZE(sones)];

		cblas_scopy(ARRAY_SIZE(sones), sones, 1, sr, 1);
		cblas_sscal(ARRAY_SIZE(sr), 1.0, sr, 1);
		TEST(EQUAL_ARRAY(sr, sones));

	SUITE("cblas_dscal");

		double dr[ARRAY_SIZE(dones)];

		cblas_dcopy(ARRAY_SIZE(dones), dones, 1, dr, 1);
		cblas_dscal(ARRAY_SIZE(dr), 1.0, dr, 1);
		TEST(EQUAL_ARRAY(dr, dones));
}

//------------------------------------------------------
// test cases for s/drot()
//------------------------------------------------------
static void test_rot(void)
{
	SUITE("cblas_srot");
	{
		float angle = (float)M_PI_2;
		float a[] = { 1.0f, 0.0f };
		float b[] = { 0.0f, 1.0f };
		float c[] = { 0.0f, 1.0f };
		float d[] = { -1.0f, 0.0f };

		float cosine = (float)cos(angle);
		float sine = (float)sin(angle);

		cblas_srot(2, a, 1, b, 1, cosine, sine);
		TEST(equal_sarray_epsilon(a, c, ARRAY_SIZE(a)));
		TEST(equal_sarray_epsilon(b, d, ARRAY_SIZE(b)));
	}

	{
		float angle = (float)M_PI_4;
		float a[] = { 1.0f, 0.0f };
		float b[] = { 0.0f, 1.0f };
		float c[] = { 0.7071f, 0.7071f };
		float d[] = { -0.7071f, 0.7071f };

		float cosine = (float)cos(angle);
		float sine = (float)sin(angle);

		cblas_srot(2, a, 1, b, 1, cosine, sine);
		TEST(equal_sarray_epsilon(a, c, ARRAY_SIZE(a)));
		TEST(equal_sarray_epsilon(b, d, ARRAY_SIZE(b)));
	}

	SUITE("cblas_drot");
	{
		double angle = M_PI_2;
		double a[] = { 1.0, 0.0 };
		double b[] = { 0.0, 1.0 };
		double c[] = { 0.0, 1.0 };
		double d[] = { -1.0, 0.0 };

		double cosine = cos(angle);
		double sine = sin(angle);

		cblas_drot(2, a, 1, b, 1, cosine, sine);
		TEST(equal_darray_epsilon(a, c, ARRAY_SIZE(a)));
		TEST(equal_darray_epsilon(b, d, ARRAY_SIZE(b)));
	}

	{
		double angle = M_PI_4;
		double a[] = { 1.0, 0.0 };
		double b[] = { 0.0, 1.0 };
		double c[] = { 0.7071, 0.7071 };
		double d[] = { -0.7071, 0.7071 };

		double cosine = cos(angle);
		double sine = sin(angle);

		cblas_drot(2, a, 1, b, 1, cosine, sine);
		TEST(equal_darray_epsilon(a, c, ARRAY_SIZE(a)));
		TEST(equal_darray_epsilon(b, d, ARRAY_SIZE(b)));
	}
}

//------------------------------------------------------
// test cases for s/dasum()
//------------------------------------------------------
static void test_asum(void)
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
		TEST(dr == ARRAY_SIZE(dones) * 1.0);

		dr = cblas_dasum(ARRAY_SIZE(da), da, 1);
		TEST(dr == 45.0);
}

//------------------------------------------------------
// test cases for s/dnrm2()
//------------------------------------------------------
static void test_nrm2(void)
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
static void test_ger(void)
{
	SUITE("cblas_sger");
		{
			float sx[3] = {1,1,1};
			float sy[3] = {1,1,1};
			float sa[9] = {0,0,0,0,0,0,0,0,0};
			float sr[9] = {1,1,1,1,1,1,1,1,1};

			cblas_sger(CblasRowMajor, 3, 3, 1.0f, sx, 1, sy, 1, sa, 3);
			TEST(EQUAL_ARRAY(sa, sr));

			cblas_sscal(9, 3.0f, sr, 1);
			cblas_sger(CblasColMajor, 3, 3, 2.0f, sx, 1, sy, 1, sa, 3);
			TEST(EQUAL_ARRAY(sa, sr));
		}

		{
			float sx[4] = {1,1,1,1};
			float sy[4] = {1,1,1,1};
			float sa[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
			float sr[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

			cblas_sger(CblasRowMajor, 4, 4, 1.0f, sx, 1, sy, 1, sa, 4);
			TEST(EQUAL_ARRAY(sa, sr));
		}

		{
			float sx[5] = {1,1,1,1,1};
			float sy[5] = {1,1,1,1,1};
			float sa[25];
			float sr[25];

			cblas_ssetv(25, sa, 0.0f);
			cblas_ssetv(25, sr, 1.0f);

			cblas_sger(CblasRowMajor, 5, 5, 1.0f, sx, 1, sy, 1, sa, 5);
//print_sarray(25, sa);
//print_sarray(25, sr);
			TEST(EQUAL_ARRAY(sa, sr));
		}

		{
			float sx[8] = { 1,1,1,1,1,1,1,1 };
			float sy[8] = { 1,1,1,1,1,1,1,1 };
			float sa[64];
			float sr[64];

			cblas_ssetv(64, sa, 0.0f);
			cblas_ssetv(64, sr, 1.0f);

			cblas_sger(CblasRowMajor, 8, 8, 1.0f, sx, 1, sy, 1, sa, 8);
			//print_sarray(25, sa);
			//print_sarray(25, sr);
			TEST(EQUAL_ARRAY(sa, sr));
		}

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
static void test_gemv(void)
{
	SUITE("cblas_sgemv");

		float smtx[9] = {1,0,0,0,1,0,0,0,1};
		float sx[3] = {2,2,2};
		// float sy[3] = {0,0,0};
		float sz[3] = {4, 4, 4};

		TEST(cbu_sge_is_identity(smtx, 3, 3));

		{
			float sy[3] = {0};
			cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0f, smtx, 3, sx, 1, 1.0f, sy, 1);
			TEST(EQUAL_ARRAY(sx, sy));
		}

		{
			float sy[3] = {0};
			cblas_sgemv(CblasRowMajor, CblasNoTrans, 3, 3, 2.0f, smtx, 3, sx, 1, 1.0f, sy, 1);
			TEST(EQUAL_ARRAY(sz, sy));
		}

		{
			float sy[3] = {0};
			cblas_sgemv(CblasColMajor, CblasTrans, 3, 3, 1.0f, smtx, 3, sx, 1, 1.0f, sy, 1);
			TEST(EQUAL_ARRAY(sx, sy));
		}

	SUITE("cblas_dgemv");

		double dmtx[9] = {1,0,0,0,1,0,0,0,1};
		double dx[3] = {2,2,2};
		// double dy[3] = {0,0,0};
		double dz[3] = {4,4,4};

		TEST(cbu_dge_is_identity(dmtx, 3, 3));

		{
			double dy[3] = {0};
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 1.0, dmtx, 3, dx, 1, 1.0, dy, 1);
			TEST(EQUAL_ARRAY(dx, dy));
		}

		{
			double dy[3] = {0};
			cblas_dgemv(CblasRowMajor, CblasNoTrans, 3, 3, 2.0, dmtx, 3, dx, 1, 1.0, dy, 1);
			TEST(EQUAL_ARRAY(dz, dy));
		}

		{
			double dy[3] = {0};
			cblas_dgemv(CblasColMajor, CblasTrans, 3, 3, 1.0, dmtx, 3, dx, 1, 1.0, dy, 1);
			TEST(EQUAL_ARRAY(dx, dy));
		}
}

//------------------------------------------------------
// test cases for s/drotg()
//------------------------------------------------------
static void test_rotg(void)
{
	SUITE("cblas_srotg");

	// Test 1: Standard case (b > a)
	{
		float a = 3.0f;
		float b = 4.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0.6, s = 0.8
		TEST(EQUAL_EPSILON(a, 5.0f));           // a should contain r
		TEST(EQUAL_EPSILON(c, 0.6f));           // c = 3/5 = 0.6
		TEST(EQUAL_EPSILON(s, 0.8f));           // s = 4/5 = 0.8
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0f));   // Verify c^2 + s^2 = 1
	}

	// Test 2: Standard case (a > b)
	{
		float a = 4.0f;
		float b = 3.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0.8, s = 0.6
		TEST(EQUAL_EPSILON(a, 5.0f));           // a should contain r
		TEST(EQUAL_EPSILON(c, 0.8f));           // c = 4/5 = 0.8
		TEST(EQUAL_EPSILON(s, 0.6f));           // s = 3/5 = 0.6
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0f));   // Verify c^2 + s^2 = 1
	}

	// Test 3: Zero input case
	{
		float a = 0.0f;
		float b = 0.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// Expected: r = 0, c = 1, s = 0
		TEST(EQUAL_EPSILON(a, 0.0f));
		TEST(EQUAL_EPSILON(b, 0.0f));
		TEST(EQUAL_EPSILON(c, 1.0f));
		TEST(EQUAL_EPSILON(s, 0.0f));
	}

	// Test 4: Only a is zero
	{
		float a = 0.0f;
		float b = 5.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0, s = 1
		TEST(EQUAL_EPSILON(a, 5.0f));
		TEST(EQUAL_EPSILON(c, 0.0f));
		TEST(EQUAL_EPSILON(s, 1.0f));
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0f));
	}

	// Test 5: Only b is zero
	{
		float a = 5.0f;
		float b = 0.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 1, s = 0
		TEST(EQUAL_EPSILON(a, 5.0f));
		TEST(EQUAL_EPSILON(c, 1.0f));
		TEST(EQUAL_EPSILON(s, 0.0f));
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0f));
	}

	// Test 6: Negative values
	{
		float a = -3.0f;
		float b = 4.0f;
		float c, s;
		
		cblas_srotg(&a, &b, &c, &s);
		
		// r should have sign of larger magnitude value
		TEST(EQUAL_EPSILON(fabsf(a), 5.0f));    // |r| = 5
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0f));   // Verify c^2 + s^2 = 1
	}

	SUITE("cblas_drotg");

	// Test 1: Standard case (b > a)
	{
		double a = 3.0;
		double b = 4.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0.6, s = 0.8
		TEST(EQUAL_EPSILON(a, 5.0));            // a should contain r
		TEST(EQUAL_EPSILON(c, 0.6));            // c = 3/5 = 0.6
		TEST(EQUAL_EPSILON(s, 0.8));            // s = 4/5 = 0.8
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0));    // Verify c^2 + s^2 = 1
	}

	// Test 2: Standard case (a > b)
	{
		double a = 4.0;
		double b = 3.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0.8, s = 0.6
		TEST(EQUAL_EPSILON(a, 5.0));            // a should contain r
		TEST(EQUAL_EPSILON(c, 0.8));            // c = 4/5 = 0.8
		TEST(EQUAL_EPSILON(s, 0.6));            // s = 3/5 = 0.6
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0));    // Verify c^2 + s^2 = 1
	}

	// Test 3: Zero input case
	{
		double a = 0.0;
		double b = 0.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// Expected: r = 0, c = 1, s = 0
		TEST(EQUAL_EPSILON(a, 0.0));
		TEST(EQUAL_EPSILON(b, 0.0));
		TEST(EQUAL_EPSILON(c, 1.0));
		TEST(EQUAL_EPSILON(s, 0.0));
	}

	// Test 4: Only a is zero
	{
		double a = 0.0;
		double b = 5.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 0, s = 1
		TEST(EQUAL_EPSILON(a, 5.0));
		TEST(EQUAL_EPSILON(c, 0.0));
		TEST(EQUAL_EPSILON(s, 1.0));
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0));
	}

	// Test 5: Only b is zero
	{
		double a = 5.0;
		double b = 0.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// Expected: r = 5.0, c = 1, s = 0
		TEST(EQUAL_EPSILON(a, 5.0));
		TEST(EQUAL_EPSILON(c, 1.0));
		TEST(EQUAL_EPSILON(s, 0.0));
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0));
	}

	// Test 6: Negative values
	{
		double a = -3.0;
		double b = 4.0;
		double c, s;
		
		cblas_drotg(&a, &b, &c, &s);
		
		// r should have sign of larger magnitude value
		TEST(EQUAL_EPSILON(fabs(a), 5.0));      // |r| = 5
		TEST(EQUAL_EPSILON(c*c + s*s, 1.0));    // Verify c^2 + s^2 = 1
	}

}

//------------------------------------------------------
// test cases for s/dgemm()
//------------------------------------------------------
static void test_gemm(void)
{
	SUITE("cblas_sgemm");

#if 1
#define S 4
	float samtx[16] = {
		1, 2, 3, 4, 
		5, 6, 7, 8, 
		9, 10, 11, 12,
		13, 14, 15, 16
	};

	float sbmtx[16] = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};
#else
#define S 5
	float samtx[] = {
		1, 2, 3, 4, 5,
		6, 7, 8, 9, 10,
		11, 12, 13, 14, 15,
		16, 17, 18, 19, 20,
		21, 22, 23, 24, 25
	};

	float sbmtx[] = {
		1,0,0,0,0,
		0,1,0,0,0,
		0,0,1,0,0,
		0,0,0,1,0,
		0,0,0,0,1
	};
#endif

	{
		float scmtx[25] = { 0 };
		// 	1,1,1,1,
		// 	1,1,1,1,
		// 	1,1,1,1,
		// 	1,1,1,1
		// };

		TEST(cbu_sge_is_identity(sbmtx, S, S));

		// print_sarray(16, samtx);
		// print_sarray(16, scmtx);
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, S, S, 1.0f, samtx, S, sbmtx, S, 1.0f, scmtx, S);
		//print_sarray(25, scmtx);

		TEST(EQUAL_ARRAY(samtx, scmtx));
	}

	SUITE("cblas_dgemm");

	double damtx[16] = {
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
		13, 14, 15, 16
	};
	
	double dbmtx[16] = {
		1,0,0,0,
		0,1,0,0,
		0,0,1,0,
		0,0,0,1
	};

	double dcmtx[16] = { 0 };

	TEST(cbu_dge_is_identity(dbmtx, 4, 4));

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 4, 4, 4, 1.0, damtx, 4, dbmtx, 4, 1.0, dcmtx, 4);

	TEST(EQUAL_ARRAY(damtx, dcmtx));

	// Test with non-multiple of 4 size to ensure AddDot is called
	SUITE("cblas_sgemm (non-mult-4)");
	{
		#define N5 5
		float a5x5[25] = {
			1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15,
			16, 17, 18, 19, 20,
			21, 22, 23, 24, 25
		};

		float b5x5_identity[25] = {
			1,0,0,0,0,
			0,1,0,0,0,
			0,0,1,0,0,
			0,0,0,1,0,
			0,0,0,0,1
		};

		float c5x5[25] = { 0 };

		TEST(cbu_sge_is_identity(b5x5_identity, N5, N5));

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N5, N5, N5, 1.0f, a5x5, N5, b5x5_identity, N5, 1.0f, c5x5, N5);

		TEST(EQUAL_ARRAY_SIZE(a5x5, c5x5, 25));
	}
}

//------------------------------------------------------
// BLAS level 1 testing
//------------------------------------------------------
static void test_level1(void)
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
// BLAS level 2 testing
//------------------------------------------------------
static void test_level2(void)
{
	MODULE("BLAS Level2");

	test_ger();
	test_gemv();
}

//------------------------------------------------------
// BLAS level 3 testing
//------------------------------------------------------
static void test_level3(void)
{
	MODULE("BLAS Level3");

	test_gemm();
}

//------------------------------------------------------
//
//------------------------------------------------------
int test_main(int argc, char *argv[])
{
	(void)argc;
	(void)argv;
	sbig_ones	= svec_ones(BIG_ARRAY);
	sbig_zeroes	= svec_zeroes(BIG_ARRAY);

	dbig_ones	= dvec_ones(BIG_ARRAY);
	dbig_zeroes = dvec_zeroes(BIG_ARRAY);

	cblas_init(CBLAS_DEFAULT_THREADS);

	//cblas_set_num_threads(2);
	//cblas_set_num_threads(10);

	cblas_print_configuration();
	
	test_level1();
	test_level2();
	test_level3();

	free(sbig_ones);
	free(sbig_zeroes);

	free(dbig_ones);
	free(dbig_zeroes);

	// Shutdown CBLAS library
	cblas_shutdown();

	return 0;
}
