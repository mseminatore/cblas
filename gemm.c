//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include <stdio.h>

// macros to simpify matrix element access
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incx]

//------------------------------------------------------
// compute dot product of row of X and col of Y
//------------------------------------------------------
void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *gamma)
{
	for (int p = 0; p < k; p++) {
		*gamma += x[p] * Y(p);
	}
}

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows
//------------------------------------------------------
void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    register float 
        c_00, c_10, c_20, c_30,
        c_01, c_11, c_21, c_31,
        c_02, c_12, c_22, c_32,
        c_03, c_13, c_23, c_33;
    register float a_p0, a_p1, a_p2, a_p3;
    register float b_0p, b_1p, b_2p, b_3p;
    float *a_p0_ptr, *a_p1_ptr, *a_p2_ptr, *a_p3_ptr;

    a_p0_ptr = &A(0, 0);
    a_p1_ptr = &A(0, 1);
    a_p2_ptr = &A(0, 2);
    a_p3_ptr = &A(0, 3);
    
    c_00 = 0.0f; c_10 = 0.0f; c_20 = 0.0f; c_30 = 0.0f;
    c_01 = 0.0f; c_11 = 0.0f; c_21 = 0.0f; c_31 = 0.0f;
    c_02 = 0.0f; c_12 = 0.0f; c_22 = 0.0f; c_32 = 0.0f;
    c_03 = 0.0f; c_13 = 0.0f; c_23 = 0.0f; c_33 = 0.0f;

	for (int p = 0; p < k; p++) {
        a_p0 = *a_p0_ptr++;
        a_p1 = *a_p1_ptr++;
        a_p2 = *a_p2_ptr++;
        a_p3 = *a_p3_ptr++;

        b_0p = B(0, p);
        b_1p = B(1, p);
        b_2p = B(2, p);
        b_3p = B(3, p);

        // row 1 and row 2
        c_00 += a_p0 * b_0p;
        c_01 += a_p1 * b_0p;

        c_10 += a_p0 * b_1p;
        c_11 += a_p1 * b_1p;

        c_20 += a_p0 * b_2p;
        c_21 += a_p1 * b_2p;

        c_30 += a_p0 * b_3p;
        c_31 += a_p1 * b_3p;

        // row 3 and row 4
        c_02 += a_p2 * b_0p;
        c_03 += a_p3 * b_0p;

        c_12 += a_p2 * b_1p;
        c_13 += a_p3 * b_1p;

        c_22 += a_p2 * b_2p;
        c_23 += a_p3 * b_2p;

        c_32 += a_p2 * b_3p;
        c_33 += a_p3 * b_3p;
    }

    C(0, 0) = c_00; C(1, 0) = c_10; C(2, 0) = c_20; C(3,0) = c_30;
    C(0, 1) = c_01; C(1, 1) = c_11; C(2, 1) = c_21; C(3,1) = c_31;
    C(0, 2) = c_02; C(1, 2) = c_12; C(2, 2) = c_22; C(3,2) = c_32;
    C(0, 3) = c_03; C(1, 3) = c_13; C(2, 3) = c_23; C(3,3) = c_33;
}

//------------------------------------------------------
// compute 4 dot products at a time
// 4 rows of A by 1 column of B
//------------------------------------------------------
void AddDot1x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    register float c_00, c_01, c_02, c_03, b_0p;
    float *a0, *a1, *a2, *a3;

    // grab the start of 4 rows of A
    a0 = &A(0, 0);
    a1 = &A(0, 1);
    a2 = &A(0, 2);
    a3 = &A(0, 3);

    c_00 = 0.0f;
    c_01 = 0.0f;
    c_02 = 0.0f;
    c_03 = 0.0f;

    // 
    for (int p = 0; p < k; p += 4) {
        b_0p = B(0, p);

        c_00 += *a0 * b_0p;
        c_01 += *a1 * b_0p;
        c_02 += *a2 * b_0p;
        c_03 += *a3 * b_0p;

        b_0p = B(0, p + 1);

        c_00 += *(a0+1) * b_0p;
        c_01 += *(a1+1) * b_0p;
        c_02 += *(a2+1) * b_0p;
        c_03 += *(a3+1) * b_0p;

        b_0p = B(0, p + 2);

        c_00 += *(a0+2) * b_0p;
        c_01 += *(a1+2) * b_0p;
        c_02 += *(a2+2) * b_0p;
        c_03 += *(a3+2) * b_0p;

        b_0p = B(0, p + 3);

        c_00 += *(a0+3) * b_0p;
        c_01 += *(a1+3) * b_0p;
        c_02 += *(a2+3) * b_0p;
        c_03 += *(a3+3) * b_0p;

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
    }

    C(0,0) += c_00;
    C(0,1) += c_01;
    C(0,2) += c_02;
    C(0,3) += c_03;
}

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row += 4)    // loop over rows of C unrolled by 4
        for (int col = 0; col < n; col += 4)   // loop over cols of C
        {
            AddDot4x4(k, &A(0, row), lda, &B(col, 0), ldb, &C(col, row), ldc);
        }

    // TODO - handle remainder for matrices that are not multiples of 4 in size!
}

//------------------------------------------------------
// single-precision reference matrix multipl
//------------------------------------------------------
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

//------------------------------------------------------
// double-precision general matrix multiply
//------------------------------------------------------
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

