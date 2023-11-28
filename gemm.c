//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

// macros to simpify matrix element access
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//
void AddDot1x4(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *a, CBLAS_INDEX lda)
{
	for (int p = 0; p < k; p++)
	{
		A(0, 0) += *x * y[p];
		A(1, 0) += *x * y[p + 1];
		A(2, 0) += *x * y[p + 2];
		A(3, 0) += *x * y[p + 3];
	}
}

#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incx]

// compute dot product of row of X and col of Y
void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *gamma)
{
	for (int p = 0; p < k; p++)
	{
		*gamma += x[p] * Y(p);
	}
}

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc)
{
    for (int col = 0; col < n; col += 4)
        for (int row = 0; row < m; row++)
        {
            AddDot(k, &A(0, row), lda, &B(col, 0), &C(col, row));
            AddDot(k, &A(0, row), lda, &B(col + 1, 0), &C(col + 1, row));
            AddDot(k, &A(0, row), lda, &B(col + 2, 0), &C(col + 2, row));
            AddDot(k, &A(0, row), lda, &B(col + 3, 0), &C(col + 3, row));
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

