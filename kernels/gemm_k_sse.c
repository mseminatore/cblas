//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - SSE implementation for x86-64 (128-bit, non-FMA)
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product of row of X and col of Y (scalar fallback)
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
// compute 16 dot products at a time, 4 cols x 4 rows (SSE 128-bit)
// alpha is applied when storing results back to C
//------------------------------------------------------
static void AddDot4x4_sse(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc, float alpha)
{
    (void)lda;
    (void)ldb;
    __m128 c_row1, c_row2, c_row3, c_row4;
    __m128 c_old1, c_old2, c_old3, c_old4;
    __m128 b_row;
    __m128 a_p0, a_p1, a_p2, a_p3;
    __m128 alpha_vec = _mm_set1_ps(alpha);
    
    // Load current C values
    c_old1 = _mm_loadu_ps(&C(0,0));
    c_old2 = _mm_loadu_ps(&C(0,1));
    c_old3 = _mm_loadu_ps(&C(0,2));
    c_old4 = _mm_loadu_ps(&C(0,3));
    
    // Initialize accumulators to zero
    c_row1 = _mm_setzero_ps();
    c_row2 = _mm_setzero_ps();
    c_row3 = _mm_setzero_ps();
    c_row4 = _mm_setzero_ps();

    for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // load and duplicate 
        a_p0 = _mm_load_ps1(a);
        a_p1 = _mm_load_ps1(a + 1);
        a_p2 = _mm_load_ps1(a + 2);
        a_p3 = _mm_load_ps1(a + 3);

        // Prefetch data ahead
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * 4), 0, 3);
        }

        a += 4;

        // Use unaligned load for b since alignment is not guaranteed
        b_row = _mm_loadu_ps(b);

        b += 4;

        // rows 1 - 4 using SSE
        c_row1 = _mm_add_ps(c_row1, _mm_mul_ps(a_p0, b_row));
        c_row2 = _mm_add_ps(c_row2, _mm_mul_ps(a_p1, b_row));
        c_row3 = _mm_add_ps(c_row3, _mm_mul_ps(a_p2, b_row));
        c_row4 = _mm_add_ps(c_row4, _mm_mul_ps(a_p3, b_row));
    }

    // Apply alpha and accumulate: C = C + alpha * (A*B)
    c_row1 = _mm_add_ps(c_old1, _mm_mul_ps(alpha_vec, c_row1));
    c_row2 = _mm_add_ps(c_old2, _mm_mul_ps(alpha_vec, c_row2));
    c_row3 = _mm_add_ps(c_old3, _mm_mul_ps(alpha_vec, c_row3));
    c_row4 = _mm_add_ps(c_old4, _mm_mul_ps(alpha_vec, c_row4));

    // Store results
    _mm_storeu_ps(&C(0, 0), c_row1);
    _mm_storeu_ps(&C(0, 1), c_row2);
    _mm_storeu_ps(&C(0, 2), c_row3);
    _mm_storeu_ps(&C(0, 3), c_row4);
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
// PackMatrixA_trans - Pack 4×k panel of A from transposed storage
// A^T[i,p] = a[p*lda+i], so row i of logical A is column i of stored a
//------------------------------------------------------
static void PackMatrixA_trans(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
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
// PackMatrixB_trans - Pack k×4 panel of B from transposed storage
// B^T[p,j] = b[j*ldb+p], so column j of logical B is row j of stored b
//------------------------------------------------------
static void PackMatrixB_trans(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
{
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
// InnerKernel - SSE implementation (128-bit)
// Correct GotoBLAS-style packing: pack once per mc×kc tile, then iterate
//------------------------------------------------------
// InnerKernel - SSE 128-bit optimized inner kernel with 4x4 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_sse(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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

    CBLAS_INDEX a_row_stride = transA ? 1 : lda;
    CBLAS_INDEX a_col_stride = transA ? lda : 1;
    CBLAS_INDEX b_row_stride = transB ? 1 : ldb;
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    CBLAS_INDEX row, col;

    for (row = 0; row + 4 <= m; row += 4)
    {
        float *a_row_ptr = a + row * a_row_stride;

        // Pack this 4×k panel of A once
        if (transA)
            PackMatrixA_trans(k, a_row_ptr, lda, packedA);
        else
            PackMatrixA(k, a_row_ptr, lda, packedA);

        for (col = 0; col + 4 <= n; col += 4)
        {
            float *b_col_ptr = b + col * b_col_stride;

            // Pack each 4-column panel of B for this iteration
            if (transB)
                PackMatrixB_trans(k, b_col_ptr, ldb, packedB);
            else
                PackMatrixB(k, b_col_ptr, ldb, packedB);

            AddDot4x4_sse(k, packedA, 4, packedB, 4, &C(col, row), ldc, alpha);
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
// SGEMM kernel - SSE version (128-bit)
//------------------------------------------------------
void sgemm_k_sse(cblas_args_t* args)
{
    InnerKernel_sse(args->ib, args->n, args->pb, 
                    args->a, args->lda, 
                    args->b, args->ldb, 
                    args->c, args->ldc,
                    args->alpha_s, args->thread_id,
                    args->transa, args->transb);
}

#endif // x86_64
