//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_sger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, float alpha, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *a, CBLAS_INDEX lda)
{
    if (layout == CblasRowMajor)
    {
	 if (alpha == 1.0f)
	 {
	 //	for (int a_col = 0; a_col < a->cols; a_col++)
	 //	{
	 //		for (int b_col = 0; b_col < b->cols; b_col++)
	 //		{
	 //			dest->values[a_col * b->cols + b_col] += a->values[a_col] * b->values[b_col];
	 //		}
	 //	}
	 //}
	 //else
	 //{
	 //	for (int a_col = 0; a_col < a->cols; a_col++)
	 //	{
	 //		for (int b_col = 0; b_col < b->cols; b_col++)
	 //		{
	 //			dest->values[a_col * b->cols + b_col] += alpha * a->values[a_col] * b->values[b_col];
	 //		}
	 //	}
	 }
    } else
    {
		assert(0 && "Error: not yet implemented.");
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
void cblas_dger(CBLAS_LAYOUT layout, CBLAS_INDEX m, CBLAS_INDEX n, double alpha, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *a, CBLAS_INDEX lda)
{

}
