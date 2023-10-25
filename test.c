//------------------------------------------------------
//
// Copyright 2022 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include "test.h"
#include "cblas.h"

//
int test_number = 0;
int test_failures = 0;
int test_suites = 0;

static float szeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static float sones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static float sa[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sb[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static float sc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static float sd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

static double dzeros[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
static double dones[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
static double da[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double db[] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
static double dc[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};
static double dd[] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0};

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
	TEST(0.0f == dr);

	dr = cblas_ddot(ARRAY_SIZE(dzeros), dzeros, 1, dzeros, 1);
	TEST(0.0f == dr);

	dr = cblas_ddot(ARRAY_SIZE(dones), dones, 1, dones, 1);
	TEST(ARRAY_SIZE(dones) == dr);

}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_copy()
{
	SUITE("cblas_copy");

	float sr[ARRAY_SIZE(sones)];

	cblas_scopy(ARRAY_SIZE(sones), sones, 1, sr, 1);
	TEST(EQUAL_ARRAY(sones, sr));

	double dr[ARRAY_SIZE(dones)];

	cblas_dcopy(ARRAY_SIZE(dones), dones, 1, dr, 1);
	TEST(EQUAL_ARRAY(dones, dr));
}

//------------------------------------------------------
//
//------------------------------------------------------
static void test_axpy()
{
	SUITE("cblas_axpy");

}

//------------------------------------------------------
//
//------------------------------------------------------
int main(int argc, char *argv[])
{
	BEGIN_TESTS();

	test_swap();
	test_dot();
	test_copy();
	test_axpy();

	END_TESTS();
}