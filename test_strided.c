//------------------------------------------------------
//
// test_strided.c - Strided operation tests for BLAS Level 1
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "test.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

#define STRIDED_ARRAY_SIZE 20
#define LOGICAL_SIZE 10  // With stride=2, need 20 elements for 10 logical elements

// Test arrays
static float sa[STRIDED_ARRAY_SIZE];
static float sb[STRIDED_ARRAY_SIZE];
static float sc[STRIDED_ARRAY_SIZE];
static float sd[STRIDED_ARRAY_SIZE];
static float sresult[STRIDED_ARRAY_SIZE];

static double da[STRIDED_ARRAY_SIZE];
static double db[STRIDED_ARRAY_SIZE];
static double dc[STRIDED_ARRAY_SIZE];
static double dd[STRIDED_ARRAY_SIZE];
static double dresult[STRIDED_ARRAY_SIZE];

// Initialize array with pattern: 0, skip, 1, skip, 2, skip, ...
static void init_strided_float(float *arr, CBLAS_INDEX n, CBLAS_INDEX inc, float value)
{
    memset(arr, 0, STRIDED_ARRAY_SIZE * sizeof(float));
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        arr[i * inc] = value + (float)i;
    }
}

static void init_strided_double(double *arr, CBLAS_INDEX n, CBLAS_INDEX inc, double value)
{
    memset(arr, 0, STRIDED_ARRAY_SIZE * sizeof(double));
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        arr[i * inc] = value + (double)i;
    }
}

// Check if strided arrays are equal
static int equal_strided_float(const float *a, const float *b, CBLAS_INDEX n, CBLAS_INDEX inc)
{
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        if (fabsf(a[i * inc] - b[i * inc]) > 1e-5f)
            return 0;
    }
    return 1;
}

static int equal_strided_double(const double *a, const double *b, CBLAS_INDEX n, CBLAS_INDEX inc)
{
    for (CBLAS_INDEX i = 0; i < n; i++)
    {
        if (fabs(a[i * inc] - b[i * inc]) > 1e-10)
            return 0;
    }
    return 1;
}

//------------------------------------------------------
// Test cblas_scopy with stride=2
//------------------------------------------------------
static void test_scopy_stride2(void)
{
    SUITE("cblas_scopy (stride=2)...\n");
    
    // Initialize: sa = [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ...]
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);
    memset(sb, 0, sizeof(sb));
    
    // Copy with stride=2 for both (copies FROM sa TO sb)
    cblas_scopy(LOGICAL_SIZE, sa, 2, sb, 2);
    
    // Verify the copy worked
    TEST(equal_strided_float(sa, sb, LOGICAL_SIZE, 2));
}

//------------------------------------------------------
// Test cblas_dcopy with stride=2
//------------------------------------------------------
static void test_dcopy_stride2(void)
{
    SUITE("cblas_dcopy (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    memset(db, 0, sizeof(db));
    
    cblas_dcopy(LOGICAL_SIZE, da, 2, db, 2);
    
    TEST(equal_strided_double(da, db, LOGICAL_SIZE, 2));
}

//------------------------------------------------------
// Test cblas_scopy with stride=3
//------------------------------------------------------
static void test_scopy_stride3(void)
{
    SUITE("cblas_scopy (stride=3)...\n");
    
    // For stride=3, need 3*LOGICAL_SIZE elements
    float sa3[60] = {0};
    float sb3[60] = {0};
    
    init_strided_float(sa3, LOGICAL_SIZE, 3, 1.0f);
    cblas_scopy(LOGICAL_SIZE, sa3, 3, sb3, 3);
    
    TEST(equal_strided_float(sa3, sb3, LOGICAL_SIZE, 3));
}

//------------------------------------------------------
// Test cblas_dcopy with stride=3
//------------------------------------------------------
static void test_dcopy_stride3(void)
{
    SUITE("cblas_dcopy (stride=3)...\n");
    
    double da3[60] = {0};
    double db3[60] = {0};
    
    init_strided_double(da3, LOGICAL_SIZE, 3, 1.0);
    cblas_dcopy(LOGICAL_SIZE, da3, 3, db3, 3);
    
    TEST(equal_strided_double(da3, db3, LOGICAL_SIZE, 3));
}

//------------------------------------------------------
// Test cblas_sswap with stride=2
//------------------------------------------------------
static void test_sswap_stride2(void)
{
    SUITE("cblas_sswap (stride=2)...\n");
    
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);  // 1,2,3,4,5...
    init_strided_float(sb, LOGICAL_SIZE, 2, 10.0f); // 10,11,12,13,14...
    memcpy(sc, sa, sizeof(sa));
    memcpy(sd, sb, sizeof(sb));
    
    cblas_sswap(LOGICAL_SIZE, sa, 2, sb, 2);
    
    TEST(equal_strided_float(sa, sd, LOGICAL_SIZE, 2));
    TEST(equal_strided_float(sb, sc, LOGICAL_SIZE, 2));
}

//------------------------------------------------------
// Test cblas_dswap with stride=2
//------------------------------------------------------
static void test_dswap_stride2(void)
{
    SUITE("cblas_dswap (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    init_strided_double(db, LOGICAL_SIZE, 2, 10.0);
    memcpy(dc, da, sizeof(da));
    memcpy(dd, db, sizeof(db));
    
    cblas_dswap(LOGICAL_SIZE, da, 2, db, 2);
    
    TEST(equal_strided_double(da, dd, LOGICAL_SIZE, 2));
    TEST(equal_strided_double(db, dc, LOGICAL_SIZE, 2));
}

//------------------------------------------------------
// Test cblas_sdot with stride=2
//------------------------------------------------------
static void test_sdot_stride2(void)
{
    SUITE("cblas_sdot (stride=2)...\n");
    
    // [1,_,2,_,3,_,4,_,5,_] dot [1,_,1,_,1,_,1,_,1,_] = 1+2+3+4+5 = 15
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);
    init_strided_float(sb, LOGICAL_SIZE, 2, 1.0f);
    
    float result = cblas_sdot(LOGICAL_SIZE, sa, 2, sb, 2);
    
    // Both arrays have [1,_,2,_,3,_,...,10,_]
    // Dot product: 1*1 + 2*2 + 3*3 + ... + 10*10 = 1 + 4 + 9 + ... + 100 = 385
    float expected = 1.0f*1.0f + 2.0f*2.0f + 3.0f*3.0f + 4.0f*4.0f + 5.0f*5.0f + 
                     6.0f*6.0f + 7.0f*7.0f + 8.0f*8.0f + 9.0f*9.0f + 10.0f*10.0f;
    
    TEST(fabsf(result - expected) < 1e-5f);
}

//------------------------------------------------------
// Test cblas_ddot with stride=2
//------------------------------------------------------
static void test_ddot_stride2(void)
{
    SUITE("cblas_ddot (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    init_strided_double(db, LOGICAL_SIZE, 2, 1.0);
    
    double result = cblas_ddot(LOGICAL_SIZE, da, 2, db, 2);
    // Both arrays have [1,_,2,_,3,_,...,10,_]
    // Dot product: 1*1 + 2*2 + ... + 10*10 = 385
    double expected = 1.0*1.0 + 2.0*2.0 + 3.0*3.0 + 4.0*4.0 + 5.0*5.0 + 
                      6.0*6.0 + 7.0*7.0 + 8.0*8.0 + 9.0*9.0 + 10.0*10.0;
    
    TEST(fabs(result - expected) < 1e-10);
}

//------------------------------------------------------
// Test cblas_saxpy with stride=2
//------------------------------------------------------
static void test_saxpy_stride2(void)
{
    SUITE("cblas_saxpy (stride=2)...\n");
    
    // y = alpha*x + y where alpha=2.0
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);  // x: 1,2,3,...,10
    init_strided_float(sb, LOGICAL_SIZE, 2, 0.0f);  // y: 0,1,2,...,9
    
    cblas_saxpy(LOGICAL_SIZE, 2.0f, sa, 2, sb, 2);
    
    // Expected: y[i] = 2*(1+i) + i = 2 + 3i
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        float expected = 2.0f * (1.0f + i) + i;
        TEST(fabsf(sb[i * 2] - expected) < 1e-5f);
    }
}

//------------------------------------------------------
// Test cblas_daxpy with stride=2
//------------------------------------------------------
static void test_daxpy_stride2(void)
{
    SUITE("cblas_daxpy (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    init_strided_double(db, LOGICAL_SIZE, 2, 0.0);
    
    cblas_daxpy(LOGICAL_SIZE, 2.0, da, 2, db, 2);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        double expected = 2.0 * (1.0 + i) + i;
        TEST(fabs(db[i * 2] - expected) < 1e-10);
    }
}

//------------------------------------------------------
// Test cblas_sscal with stride=2
//------------------------------------------------------
static void test_sscal_stride2(void)
{
    SUITE("cblas_sscal (stride=2)...\n");
    
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);  // 1,2,3,...,10
    
    cblas_sscal(LOGICAL_SIZE, 3.0f, sa, 2);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        float expected = 3.0f * (1.0f + i);
        TEST(fabsf(sa[i * 2] - expected) < 1e-5f);
    }
}

//------------------------------------------------------
// Test cblas_dscal with stride=2
//------------------------------------------------------
static void test_dscal_stride2(void)
{
    SUITE("cblas_dscal (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    
    cblas_dscal(LOGICAL_SIZE, 3.0, da, 2);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        double expected = 3.0 * (1.0 + i);
        TEST(fabs(da[i * 2] - expected) < 1e-10);
    }
}

//------------------------------------------------------
// Test cblas_saxpby with stride=2
//------------------------------------------------------
static void test_saxpby_stride2(void)
{
    SUITE("cblas_saxpby (stride=2)...\n");
    
    // y = alpha*x + beta*y
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);  // x: 1,2,3,...
    init_strided_float(sb, LOGICAL_SIZE, 2, 10.0f); // y: 10,11,12,...
    
    cblas_saxpby(LOGICAL_SIZE, 2.0f, sa, 2, 3.0f, sb, 2);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        float expected = 2.0f * (1.0f + i) + 3.0f * (10.0f + i);
        TEST(fabsf(sb[i * 2] - expected) < 1e-5f);
    }
}

//------------------------------------------------------
// Test cblas_daxpby with stride=2
//------------------------------------------------------
static void test_daxpby_stride2(void)
{
    SUITE("cblas_daxpby (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    init_strided_double(db, LOGICAL_SIZE, 2, 10.0);
    
    cblas_daxpby(LOGICAL_SIZE, 2.0, da, 2, 3.0, db, 2);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        double expected = 2.0 * (1.0 + i) + 3.0 * (10.0 + i);
        TEST(fabs(db[i * 2] - expected) < 1e-10);
    }
}

//------------------------------------------------------
// Test cblas_sasum with stride=2
//------------------------------------------------------
static void test_sasum_stride2(void)
{
    SUITE("cblas_sasum (stride=2)...\n");
    
    init_strided_float(sa, LOGICAL_SIZE, 2, -5.0f);  // -5,-4,-3,...,4
    
    float result = cblas_sasum(LOGICAL_SIZE, sa, 2);
    
    // Sum of abs(-5)+abs(-4)+...+abs(4) = 5+4+3+2+1+0+1+2+3+4 = 25
    float expected = 5.0f + 4.0f + 3.0f + 2.0f + 1.0f + 0.0f + 1.0f + 2.0f + 3.0f + 4.0f;
    
    TEST(fabsf(result - expected) < 1e-5f);
}

//------------------------------------------------------
// Test cblas_dasum with stride=2
//------------------------------------------------------
static void test_dasum_stride2(void)
{
    SUITE("cblas_dasum (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, -5.0);
    
    double result = cblas_dasum(LOGICAL_SIZE, da, 2);
    double expected = 5.0 + 4.0 + 3.0 + 2.0 + 1.0 + 0.0 + 1.0 + 2.0 + 3.0 + 4.0;
    
    TEST(fabs(result - expected) < 1e-10);
}

//------------------------------------------------------
// Test cblas_snrm2 with stride=2
//------------------------------------------------------
static void test_snrm2_stride2(void)
{
    SUITE("cblas_snrm2 (stride=2)...\n");
    
    // Create array [1,_,2,_,3,_,4,_,0,_]
    init_strided_float(sa, 4, 2, 1.0f);
    sa[8] = 0.0f;  // Fifth element
    
    float result = cblas_snrm2(5, sa, 2);
    
    // sqrt(1^2 + 2^2 + 3^2 + 4^2 + 0^2) = sqrt(30)
    float expected = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
    
    TEST(fabsf(result - expected) < 1e-5f);
}

//------------------------------------------------------
// Test cblas_dnrm2 with stride=2
//------------------------------------------------------
static void test_dnrm2_stride2(void)
{
    SUITE("cblas_dnrm2 (stride=2)...\n");
    
    init_strided_double(da, 4, 2, 1.0);
    da[8] = 0.0;
    
    double result = cblas_dnrm2(5, da, 2);
    double expected = sqrt(1.0 + 4.0 + 9.0 + 16.0);
    
    TEST(fabs(result - expected) < 1e-10);
}

//------------------------------------------------------
// Test cblas_srot with stride=2
//------------------------------------------------------
static void test_srot_stride2(void)
{
    SUITE("cblas_srot (stride=2)...\n");
    
    init_strided_float(sa, LOGICAL_SIZE, 2, 1.0f);  // 1,2,3,...
    init_strided_float(sb, LOGICAL_SIZE, 2, 0.0f);  // 0,1,2,...
    
    // Rotate by 90 degrees: c=0, s=1 gives x'=y, y'=-x
    cblas_srot(LOGICAL_SIZE, sa, 2, sb, 2, 0.0f, 1.0f);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        float expected_a = (float)i;
        float expected_b = -(1.0f + i);
        TEST(fabsf(sa[i * 2] - expected_a) < 1e-5f);
        TEST(fabsf(sb[i * 2] - expected_b) < 1e-5f);
    }
}

//------------------------------------------------------
// Test cblas_drot with stride=2
//------------------------------------------------------
static void test_drot_stride2(void)
{
    SUITE("cblas_drot (stride=2)...\n");
    
    init_strided_double(da, LOGICAL_SIZE, 2, 1.0);
    init_strided_double(db, LOGICAL_SIZE, 2, 0.0);
    
    cblas_drot(LOGICAL_SIZE, da, 2, db, 2, 0.0, 1.0);
    
    for (CBLAS_INDEX i = 0; i < LOGICAL_SIZE; i++)
    {
        double expected_a = (double)i;
        double expected_b = -(1.0 + i);
        TEST(fabs(da[i * 2] - expected_a) < 1e-10);
        TEST(fabs(db[i * 2] - expected_b) < 1e-10);
    }
}

//------------------------------------------------------
// Main test runner (called from test_main.c)
//------------------------------------------------------
int test_main(int argc, char* argv[])
{
    // Initialize CBLAS library
    cblas_init(CBLAS_DEFAULT_THREADS);
    
    cblas_print_configuration();
    
    MODULE("BLAS Level1 (Strided)");
    
    // Test with stride=2
    test_scopy_stride2();
    test_dcopy_stride2();
    test_sswap_stride2();
    test_dswap_stride2();
    test_sdot_stride2();
    test_ddot_stride2();
    test_saxpy_stride2();
    test_daxpy_stride2();
    test_sscal_stride2();
    test_dscal_stride2();
    test_saxpby_stride2();
    test_daxpby_stride2();
    test_sasum_stride2();
    test_dasum_stride2();
    test_snrm2_stride2();
    test_dnrm2_stride2();
    test_srot_stride2();
    test_drot_stride2();
    
    // Test with stride=3
    test_scopy_stride3();
    test_dcopy_stride3();
    
    // Shutdown CBLAS library
    cblas_shutdown();
    
    return 0;
}
