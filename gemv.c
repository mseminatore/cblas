//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *a, CBLAS_INDEX lda, float *x, CBLAS_INDEX incx, float beta, float *y, CBLAS_INDEX incy)
{
	// float sum;

	// if (trans == Cblas_NoTranspose)
	// {
	// 	// for each row of the matrix
	// 	for (int mtx_row = 0; mtx_row < mtx->rows; mtx_row++)
	// 	{
	// 		sum = 0.0f;

	// 		for (int mtx_col = 0; mtx_col < mtx->cols; mtx_col++)
	// 		{
	// 			sum += alpha * mtx->values[mtx_row * mtx->cols + mtx_col] * v->values[mtx_col];
	// 		}

	// 		dest->values[mtx_row] = beta * dest->values[mtx_row] + sum;
	// 	}
	// }
	// else
	// {
	// 	assert(dest->rows == 1);

	// 	for (int mtx_col = 0; mtx_col < mtx->cols; mtx_col++)
	// 	{
	// 		sum = 0.0f;

	// 		for (int mtx_row = 0; mtx_row < mtx->rows; mtx_row++)
	// 		{
	// 			sum += alpha * mtx->values[mtx_row * mtx->cols + mtx_col] * v->values[mtx_row];
	// 		}

	// 		dest->values[mtx_col] = beta * dest->values[mtx_col] + sum;
	// 	}
	// }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *a, CBLAS_INDEX lda, double *x, CBLAS_INDEX incx, double beta, double *y, CBLAS_INDEX incy)
{

}
