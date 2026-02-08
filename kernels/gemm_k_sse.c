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
                            float alpha, int thread_id)
{
    // Temporarily disabled buffer pool - always use malloc
    float* packedA;
    float* packedB;
    int use_pool = 0;
    (void)thread_id;
    
    packedA = (float*)malloc(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
    packedB = (float*)malloc(cblas_gemm_kc * 4 * sizeof(float));
    
    if (!packedA || !packedB) {
        free(packedA);
        free(packedB);
        return;
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

            AddDot4x4_sse(k, packedA, 4, packedB, 4, &C(col, row), ldc, alpha);
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
// SGEMM kernel - SSE version (128-bit)
//------------------------------------------------------
void sgemm_k_sse(cblas_args_t* args)
{
    InnerKernel_sse(args->ib, args->n, args->pb, 
                    args->a, args->lda, 
                    args->b, args->ldb, 
                    args->c, args->ldc,
                    args->alpha_s, args->thread_id);
}

#endif // x86_64
