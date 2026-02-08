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
// InnerKernel - base scalar implementation
// Correct GotoBLAS-style packing: pack once per mc×kc tile, then iterate
//------------------------------------------------------
static void InnerKernel_base(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                             float* a, CBLAS_INDEX lda, 
                             float* b, CBLAS_INDEX ldb, 
                             float* c, CBLAS_INDEX ldc,
                             float alpha, int thread_id)
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

    CBLAS_INDEX row, col;

    for (row = 0; row + 4 <= m; row += 4)
    {
        // Pack this 4×k panel of A once
        PackMatrixA(k, &A(0, row), lda, packedA);

        for (col = 0; col + 4 <= n; col += 4)
        {
            // Pack each 4-column panel of B for this iteration
            PackMatrixB(k, &B(col, 0), ldb, packedB);

            AddDot4x4_base(k, packedA, 4, packedB, 4, &C(col, row), ldc, alpha);
        }

        // handle leftover columns
        switch(n - col)
        {
            case 3:     
                AddDot(k, &A(0, row), 1, &B(col + 2, 0), ldb, &C(col + 2, row), alpha);
                AddDot(k, &A(0, row+1), 1, &B(col + 2, 0), ldb, &C(col + 2, row+1), alpha);
                AddDot(k, &A(0, row+2), 1, &B(col + 2, 0), ldb, &C(col + 2, row+2), alpha);
                AddDot(k, &A(0, row+3), 1, &B(col + 2, 0), ldb, &C(col + 2, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 2:
                AddDot(k, &A(0, row), 1, &B(col + 1, 0), ldb, &C(col + 1, row), alpha);
                AddDot(k, &A(0, row+1), 1, &B(col + 1, 0), ldb, &C(col + 1, row+1), alpha);
                AddDot(k, &A(0, row+2), 1, &B(col + 1, 0), ldb, &C(col + 1, row+2), alpha);
                AddDot(k, &A(0, row+3), 1, &B(col + 1, 0), ldb, &C(col + 1, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 1:
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
                AddDot(k, &A(0, row+1), 1, &B(col, 0), ldb, &C(col, row+1), alpha);
                AddDot(k, &A(0, row+2), 1, &B(col, 0), ldb, &C(col, row+2), alpha);
                AddDot(k, &A(0, row+3), 1, &B(col, 0), ldb, &C(col, row+3), alpha);
                CBLAS_FALLTHROUGH;
            case 0: ;
        }
    }

    // handle leftover rows
    switch(m - row)
    {
        case 3: for (col = 0; col < n; col++) AddDot(k, &A(0, row + 2), 1, &B(col, 0), ldb, &C(col, row + 2), alpha);
            CBLAS_FALLTHROUGH;
        case 2: for (col = 0; col < n; col++) AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1), alpha);
            CBLAS_FALLTHROUGH;
        case 1: for (col = 0; col < n; col++) AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
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
                     args->alpha_s, args->thread_id);
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
                                   double alpha)
{
    // Redefine macros for double
    #undef A
    #undef B
    #undef C
    #define A(col, row) a[((row) * lda + (col))]
    #define B(col, row) b[((row) * ldb + (col))]
    #define C(col, row) c[((row) * ldc + (col))]

    CBLAS_INDEX row, col;

    for (row = 0; row < m; row++)
    {
        for (col = 0; col < n; col++)
        {
            AddDot_d(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
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
                           args->alpha_d);
}
