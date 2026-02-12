//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - base scalar implementation
//------------------------------------------------------

#include "cblas.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product of row of X and col of Y (scalar)
// alpha is applied to the result before accumulating
//------------------------------------------------------
static void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *gamma, float alpha)
{
    float *px = x;
    float *py = y;
    float sum = 0.0f;
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        sum += (*px) * (*py);
        px += incx;
        py += incy;
    }
    *gamma += alpha * sum;
}

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows (scalar)
// alpha is applied when storing results back to C
//------------------------------------------------------
static void AddDot4x4_base(CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc, float alpha)
{
    register float 
        c_00, c_10, c_20, c_30,
        c_01, c_11, c_21, c_31,
        c_02, c_12, c_22, c_32,
        c_03, c_13, c_23, c_33;

    register float a_p0, a_p1, a_p2, a_p3;
    register float b_0p, b_1p, b_2p, b_3p;

    (void)lda;  // Using packed format

    c_00 = 0.0f; c_10 = 0.0f; c_20 = 0.0f; c_30 = 0.0f;
    c_01 = 0.0f; c_11 = 0.0f; c_21 = 0.0f; c_31 = 0.0f;
    c_02 = 0.0f; c_12 = 0.0f; c_22 = 0.0f; c_32 = 0.0f;
    c_03 = 0.0f; c_13 = 0.0f; c_23 = 0.0f; c_33 = 0.0f;

    for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        a_p0 = *a;
        a_p1 = *(a + 1);
        a_p2 = *(a + 2);
        a_p3 = *(a + 3);

        // Prefetch data ahead
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(&B(0, p + PREFETCH_DISTANCE), 0, 3);
        }

        a += 4;

        b_0p = B(0, p);
        b_1p = B(1, p);
        b_2p = B(2, p);
        b_3p = B(3, p);

        // row 1
        c_00 += a_p0 * b_0p;
        c_10 += a_p0 * b_1p;
        c_20 += a_p0 * b_2p;
        c_30 += a_p0 * b_3p;

        // row 2
        c_01 += a_p1 * b_0p;
        c_11 += a_p1 * b_1p;
        c_21 += a_p1 * b_2p;
        c_31 += a_p1 * b_3p;

        // row 3
        c_02 += a_p2 * b_0p;
        c_12 += a_p2 * b_1p;
        c_22 += a_p2 * b_2p;
        c_32 += a_p2 * b_3p;

        // row 4
        c_03 += a_p3 * b_0p;
        c_13 += a_p3 * b_1p;
        c_23 += a_p3 * b_2p;
        c_33 += a_p3 * b_3p;
    }

    // Apply alpha scaling and accumulate into C
    C(0, 0) += alpha * c_00; C(1, 0) += alpha * c_10; C(2, 0) += alpha * c_20; C(3, 0) += alpha * c_30;
    C(0, 1) += alpha * c_01; C(1, 1) += alpha * c_11; C(2, 1) += alpha * c_21; C(3, 1) += alpha * c_31;
    C(0, 2) += alpha * c_02; C(1, 2) += alpha * c_12; C(2, 2) += alpha * c_22; C(3, 2) += alpha * c_32;
    C(0, 3) += alpha * c_03; C(1, 3) += alpha * c_13; C(2, 3) += alpha * c_23; C(3, 3) += alpha * c_33;
}

//------------------------------------------------------
// PackMatrixB - Copy a k×4 panel of B into contiguous memory
//------------------------------------------------------
static void PackMatrixB(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(0, j);

        if (j + 8 < k) {
            CBLAS_PREFETCH(&B(0, j + 8), 0, 3);
        }

        *b_to       = *b_ij_pntr;
        *(b_to + 1) = *(b_ij_pntr + 1);
        *(b_to + 2) = *(b_ij_pntr + 2);
        *(b_to + 3) = *(b_ij_pntr + 3);

        b_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixB_trans - Pack k×4 panel of B from transposed storage
// B^T[p,j] = b[j*ldb+p], so column j of logical B is row j of stored b
//------------------------------------------------------
static void PackMatrixB_trans(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
{
    // b points to logical B[0, col_start] in transposed storage
    // Logical B[p, col+c] = b[c*ldb + p]
    float *col0 = b;
    float *col1 = b + ldb;
    float *col2 = b + 2 * ldb;
    float *col3 = b + 3 * ldb;

    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        if (p + 8 < k) {
            CBLAS_PREFETCH(col0 + 8, 0, 3);
        }

        *b_to       = col0[p];
        *(b_to + 1) = col1[p];
        *(b_to + 2) = col2[p];
        *(b_to + 3) = col3[p];

        b_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixA - Copy a 4×k panel of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_0i_pntr = &A(0,0), *a_1i_pntr = &A(0,1),
          *a_2i_pntr = &A(0,2), *a_3i_pntr = &A(0,3);

    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        if (i + 8 < k) {
            CBLAS_PREFETCH(a_0i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_1i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_2i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_3i_pntr + 8, 0, 3);
        }

        *a_to       = *a_0i_pntr++;
        *(a_to + 1) = *a_1i_pntr++;
        *(a_to + 2) = *a_2i_pntr++;
        *(a_to + 3) = *a_3i_pntr++;

        a_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixA_trans - Pack 4×k panel of A from transposed storage
// A^T[i,p] = a[p*lda+i], so row i of logical A is column i of stored a
//------------------------------------------------------
static void PackMatrixA_trans(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    // a points to logical A[row_start, 0] in transposed storage
    // Logical A[row+r, p] = a[p*lda + r]
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        float *a_col = a + p * lda;

        if (p + 8 < k) {
            CBLAS_PREFETCH(a + (p + 8) * lda, 0, 3);
        }

        *a_to       = a_col[0];
        *(a_to + 1) = a_col[1];
        *(a_to + 2) = a_col[2];
        *(a_to + 3) = a_col[3];

        a_to += 4;
    }
}

//------------------------------------------------------
// InnerKernel - base scalar implementation
// Correct GotoBLAS-style packing: pack once per mc×kc tile, then iterate
// Supports all transpose combinations via transpose-aware packing
//------------------------------------------------------
static void InnerKernel_base(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                             float* a, CBLAS_INDEX lda, 
                             float* b, CBLAS_INDEX ldb, 
                             float* c, CBLAS_INDEX ldc,
                             float alpha, int thread_id,
                             CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb)
{
    cblas_gemm_buffer_t* buf = cblas_get_gemm_buffer(thread_id);
    float* packedA;
    float* packedB;
    int use_pool = 0;
    
    if (buf) {
        packedA = buf->packedA_s;
        packedB = buf->packedB_s;
        use_pool = 1;
    } else {
        packedA = (float*)malloc(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
        packedB = (float*)malloc(cblas_gemm_kc * 4 * sizeof(float));
        if (!packedA || !packedB) {
            free(packedA);
            free(packedB);
            return;
        }
    }

    int transA = (transa == CblasTrans);
    int transB = (transb == CblasTrans);

    // Strides for accessing logical elements:
    // Logical A[i,p]: NoTrans = a[i*lda+p] (row_stride=lda, col_stride=1)
    //                 Trans   = a[p*lda+i] (row_stride=1, col_stride=lda)
    CBLAS_INDEX a_row_stride = transA ? 1 : lda;
    CBLAS_INDEX a_col_stride = transA ? lda : 1;
    // Logical B[p,j]: NoTrans = b[p*ldb+j] (row_stride=ldb, col_stride=1)
    //                 Trans   = b[j*ldb+p] (row_stride=1, col_stride=ldb)
    CBLAS_INDEX b_row_stride = transB ? 1 : ldb;
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    CBLAS_INDEX row, col;

    for (row = 0; row + 4 <= m; row += 4)
    {
        // Pointer to logical A[row, 0]
        float *a_row = a + row * a_row_stride;

        // Pack this 4×k panel of A once
        if (transA)
            PackMatrixA_trans(k, a_row, lda, packedA);
        else
            PackMatrixA(k, a_row, lda, packedA);

        for (col = 0; col + 4 <= n; col += 4)
        {
            // Pointer to logical B[0, col]
            float *b_col = b + col * b_col_stride;

            // Pack each 4-column panel of B for this iteration
            if (transB)
                PackMatrixB_trans(k, b_col, ldb, packedB);
            else
                PackMatrixB(k, b_col, ldb, packedB);

            AddDot4x4_base(k, packedA, 4, packedB, 4, &C(col, row), ldc, alpha);
        }

        // handle leftover columns
        switch(n - col)
        {
            case 3:     
                AddDot(k, a + (row)   * a_row_stride, a_col_stride, b + (col+2) * b_col_stride, b_row_stride, &C(col + 2, row), alpha);
                AddDot(k, a + (row+1) * a_row_stride, a_col_stride, b + (col+2) * b_col_stride, b_row_stride, &C(col + 2, row+1), alpha);
                AddDot(k, a + (row+2) * a_row_stride, a_col_stride, b + (col+2) * b_col_stride, b_row_stride, &C(col + 2, row+2), alpha);
                AddDot(k, a + (row+3) * a_row_stride, a_col_stride, b + (col+2) * b_col_stride, b_row_stride, &C(col + 2, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 2:
                AddDot(k, a + (row)   * a_row_stride, a_col_stride, b + (col+1) * b_col_stride, b_row_stride, &C(col + 1, row), alpha);
                AddDot(k, a + (row+1) * a_row_stride, a_col_stride, b + (col+1) * b_col_stride, b_row_stride, &C(col + 1, row+1), alpha);
                AddDot(k, a + (row+2) * a_row_stride, a_col_stride, b + (col+1) * b_col_stride, b_row_stride, &C(col + 1, row+2), alpha);
                AddDot(k, a + (row+3) * a_row_stride, a_col_stride, b + (col+1) * b_col_stride, b_row_stride, &C(col + 1, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 1:
                AddDot(k, a + (row)   * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
                AddDot(k, a + (row+1) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row+1), alpha);
                AddDot(k, a + (row+2) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row+2), alpha);
                AddDot(k, a + (row+3) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 0: ;
        }
    }

    // handle leftover rows
    switch(m - row)
    {
        case 3: for (col = 0; col < n; col++) AddDot(k, a + (row+2) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row + 2), alpha);
            CBLAS_FALLTHROUGH;
        case 2: for (col = 0; col < n; col++) AddDot(k, a + (row+1) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row + 1), alpha);
            CBLAS_FALLTHROUGH;
        case 1: for (col = 0; col < n; col++) AddDot(k, a + (row) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
            CBLAS_FALLTHROUGH;
        case 0: ;
    }
    
    if (!use_pool) {
        free(packedA);
        free(packedB);
    }
}

//------------------------------------------------------
// SGEMM kernel - base scalar version
//------------------------------------------------------
void sgemm_k_base(cblas_args_t* args)
{
    InnerKernel_base(args->ib, args->n, args->pb, 
                     args->a, args->lda, 
                     args->b, args->ldb,
                     args->c, args->ldc,
                     args->alpha_s, args->thread_id,
                     args->transa, args->transb);
}

//------------------------------------------------------
// Double-precision scalar dot product
//------------------------------------------------------
static void AddDot_d(CBLAS_INDEX k, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *gamma, double alpha)
{
    double *px = x;
    double *py = y;
    double sum = 0.0;
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        sum += (*px) * (*py);
        px += incx;
        py += incy;
    }
    *gamma += alpha * sum;
}

//------------------------------------------------------
// DGEMM base scalar kernel
// Simple implementation for CPUs without FMA
//------------------------------------------------------
static void InnerKernel_dgemm_base(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                                   double* a, CBLAS_INDEX lda, 
                                   double* b, CBLAS_INDEX ldb, 
                                   double* c, CBLAS_INDEX ldc,
                                   double alpha,
                                   CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb)
{
    // Redefine macros for double
    #undef A
    #undef B
    #undef C
    #define A(col, row) a[((row) * lda + (col))]
    #define B(col, row) b[((row) * ldb + (col))]
    #define C(col, row) c[((row) * ldc + (col))]

    int transA = (transa == CblasTrans);
    int transB = (transb == CblasTrans);

    CBLAS_INDEX a_row_stride = transA ? 1 : lda;
    CBLAS_INDEX a_col_stride = transA ? lda : 1;
    CBLAS_INDEX b_row_stride = transB ? 1 : ldb;
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    CBLAS_INDEX row, col;

    for (row = 0; row < m; row++)
    {
        for (col = 0; col < n; col++)
        {
            AddDot_d(k, a + row * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
        }
    }
}

//------------------------------------------------------
// DGEMM kernel - base scalar version
//------------------------------------------------------
void dgemm_k_base(cblas_args_t* args)
{
    InnerKernel_dgemm_base(args->ib, args->n, args->pb, 
                           (double*)args->a, args->lda, 
                           (double*)args->b, args->ldb,
                           (double*)args->c, args->ldc,
                           args->alpha_d,
                           args->transa, args->transb);
}
