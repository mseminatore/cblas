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

// compute dot product of row of X and col of Y
void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *gamma)
{
	for (int p = 0; p < k; p++) {
		*gamma += x[p] * Y(p);
	}
}

// compute 4 dot products at a time
// 4 rows of A by 1 column of B
void AddDot1x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    register float c_00, c_01, c_02, c_03, b_0p;
    float *a0, *a1, *a2, *a3;

    c_00 = 0.0f;
    c_01 = 0.0f;
    c_02 = 0.0f;
    c_03 = 0.0f;

    // grab the start of 4 rows of A
    a0 = &A(0, 0);
    a1 = &A(0, 1);
    a2 = &A(0, 2);
    a3 = &A(0, 3);

    // 
    for (int p = 0; p < k; p++) {
        b_0p = B(0, p);

        c_00 += *a0++ * b_0p;
        c_01 += *a1++ * b_0p;
        c_02 += *a2++ * b_0p;
        c_03 += *a3++ * b_0p;
    }

    // update C
    C(0, 0) += c_00;
    C(0, 1) += c_01;
    C(0, 2) += c_02;
    C(0, 3) += c_03;
}

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row += 4)    // loop over rows of C unrolled by 4
        for (int col = 0; col < n; col++)   // loop over cols of C
        {
            AddDot1x4(k, &A(0, row), lda, &B(col, 0), ldb, &C(col, row), ldc);
        }

    // TODO - handle remainder for matrices that are not multiples of 4 in size!
}

//------------------------------------------------------
// double-precision general matrix multiply
//------------------------------------------------------
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc)
{
    for (int col = 0; col < n; col++)
        for (int row = 0; row < m; row++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

