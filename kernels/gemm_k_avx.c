//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - AVX implementation for x86-64 (256-bit, non-FMA)
// Implements 4x8 micro-kernel for AVX (without FMA)
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Micro-kernel dimensions for AVX 256-bit
// 4x8 = 4 YMM registers for C (4 rows × 1 YMM of 8 floats)
#define MR 4   // Rows per micro-kernel
#define NR 8   // Columns per micro-kernel (1 YMM register of 8 floats)

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
// 4x8 micro-kernel using AVX 256-bit (non-FMA)
// Computes C[4x8] += alpha * A[4xk] * B[kx8]
// Uses 4 YMM registers for C (1 per row, 8 floats each)
//------------------------------------------------------
static void AddDot4x8_avx(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 4 rows × 1 YMM = 8 columns
    __m256 c0, c1, c2, c3;  // Rows 0-3, columns 0-7
    
    __m256 b_row;     // B row: 8 floats = 1 YMM
    __m256 a_elem;    // A element broadcast
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    // Initialize accumulators to zero
    c0 = _mm256_setzero_ps();
    c1 = _mm256_setzero_ps();
    c2 = _mm256_setzero_ps();
    c3 = _mm256_setzero_ps();
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (8 floats from packed format)
        b_row = _mm256_loadu_ps(b);
        b += NR;  // Advance to next row of packed B
        
        // Prefetch next iterations
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH_L2(a + (8 * MR));
            CBLAS_PREFETCH_L1(a + (4 * MR));
            CBLAS_PREFETCH_L2(b + (8 * NR));
            CBLAS_PREFETCH_L1(b + (4 * NR));
        }
        
        // Row 0: broadcast A[0,p] and multiply-add (non-FMA)
        a_elem = _mm256_set1_ps(a[0]);
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a_elem, b_row));
        
        // Row 1
        a_elem = _mm256_set1_ps(a[1]);
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a_elem, b_row));
        
        // Row 2
        a_elem = _mm256_set1_ps(a[2]);
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a_elem, b_row));
        
        // Row 3
        a_elem = _mm256_set1_ps(a[3]);
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a_elem, b_row));
        
        a += MR;  // Advance to next column of packed A
    }
    
    // Load old C values, apply alpha to accumulators, accumulate, store
    __m256 c_old;
    
    c_old = _mm256_loadu_ps(&C(0, 0));
    c0 = _mm256_add_ps(c_old, _mm256_mul_ps(alpha_vec, c0));
    _mm256_storeu_ps(&C(0, 0), c0);
    
    c_old = _mm256_loadu_ps(&C(0, 1));
    c1 = _mm256_add_ps(c_old, _mm256_mul_ps(alpha_vec, c1));
    _mm256_storeu_ps(&C(0, 1), c1);
    
    c_old = _mm256_loadu_ps(&C(0, 2));
    c2 = _mm256_add_ps(c_old, _mm256_mul_ps(alpha_vec, c2));
    _mm256_storeu_ps(&C(0, 2), c2);
    
    c_old = _mm256_loadu_ps(&C(0, 3));
    c3 = _mm256_add_ps(c_old, _mm256_mul_ps(alpha_vec, c3));
    _mm256_storeu_ps(&C(0, 3), c3);
}

//------------------------------------------------------
// PackMatrixA_4_trans - Pack 4×k panel of A from transposed storage
// When A is transposed, logical A[row+r, p] = a[p*lda + r]
//------------------------------------------------------
static void PackMatrixA_4_trans(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *a_col = a + i * lda;
        
        if (i + 8 < k) {
            CBLAS_PREFETCH_L2(a + (i + 8) * lda);
        }
        
        for (CBLAS_INDEX r = 0; r < MR; r++) {
            if (r < m_rows) {
                a_to[r] = a_col[r];
            } else {
                a_to[r] = 0.0f;
            }
        }
        
        a_to += MR;
    }
}

//------------------------------------------------------
// PackMatrixB_8_trans - Pack k×8 panel of B from transposed storage
// When B is transposed, logical B[p, col+c] = b[c*ldb + p]
//------------------------------------------------------
static void PackMatrixB_8_trans(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR; col++) {
            b_to[col] = b[col * ldb + j];
        }
        for (; col < NR; col++) {
            b_to[col] = 0.0f;
        }
        
        b_to += NR;
    }
}

//------------------------------------------------------
// PackMatrixB_8 - Copy a k×8 panel of B into contiguous memory
// Packing format: For each row p of B, store 8 consecutive columns
//------------------------------------------------------
static void PackMatrixB_8(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(0, j);
        
        // Prefetch ahead
        if (j + 4 < k) {
            CBLAS_PREFETCH_L2(&B(0, j + 4));
        }
        
        // Copy up to 8 columns, zero-pad if fewer
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR; col++) {
            b_to[col] = b_ij_pntr[col];
        }
        // Zero-pad remaining columns
        for (; col < NR; col++) {
            b_to[col] = 0.0f;
        }
        
        b_to += NR;
    }
}

//------------------------------------------------------
// PackMatrixA_4 - Copy a 4×k panel of A into contiguous memory
// Packing format: For each column p of A, store 4 consecutive rows
//------------------------------------------------------
static void PackMatrixA_4(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    // Handle varying number of rows (1-4)
    float *a_ptrs[4];
    
    for (CBLAS_INDEX r = 0; r < MR; r++) {
        if (r < m_rows) {
            a_ptrs[r] = &A(0, r);
        } else {
            a_ptrs[r] = NULL;  // Will be zero-padded
        }
    }
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        // Prefetch ahead
        if (i + 8 < k && a_ptrs[0]) {
            CBLAS_PREFETCH_L2(a_ptrs[0] + 8);
        }
        
        // Pack 4 rows for this column
        for (CBLAS_INDEX r = 0; r < MR; r++) {
            if (a_ptrs[r]) {
                a_to[r] = *a_ptrs[r]++;
            } else {
                a_to[r] = 0.0f;  // Zero-pad
            }
        }
        
        a_to += MR;
    }
}

//------------------------------------------------------
// InnerKernel - AVX 256-bit implementation with 4x8 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
// InnerKernel - AVX 256-bit optimized inner kernel with 4x8 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_avx(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                            float* a, CBLAS_INDEX lda, 
                            float* b, CBLAS_INDEX ldb, 
                            float* c, CBLAS_INDEX ldc,
                            float alpha, int thread_id,
                            CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb)
{
    cblas_gemm_buffer_t* buf = cblas_get_gemm_buffer(thread_id);
    float* packedA;
    float* packedB;
    int use_pool_a = 0, use_pool_b = 0;
    size_t packedB_needed = (size_t)k * n * sizeof(float);
    size_t pool_b_size = (size_t)cblas_gemm_kc * cblas_gemm_nb * sizeof(float);
    
    if (buf) {
        packedA = buf->packedA_s;
        use_pool_a = 1;
        if (packedB_needed <= pool_b_size) {
            packedB = buf->packedB_s;
            use_pool_b = 1;
        } else {
            packedB = (float*)malloc(packedB_needed);
            if (!packedB) return;
        }
    } else {
        packedA = (float*)malloc(MR * k * sizeof(float));
        packedB = (float*)malloc(packedB_needed);
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

    // Phase 1: Pack ALL B column panels once upfront
    for (col = 0; col + NR <= n; col += NR)
    {
        float *b_ptr = b + col * b_col_stride;
        if (transB)
            PackMatrixB_8_trans(k, NR, b_ptr, ldb, &packedB[col * k]);
        else
            PackMatrixB_8(k, NR, b_ptr, ldb, &packedB[col * k]);
    }

    // Phase 2: Main loop - 4 rows at a time
    for (row = 0; row + MR <= m; row += MR)
    {
        // Pack this 4×k panel of A once per row iteration
        if (transA)
            PackMatrixA_4_trans(k, MR, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_4(k, MR, a + row * a_row_stride, lda, packedA);

        // Process 8 columns at a time
        for (col = 0; col + NR <= n; col += NR)
        {
            // Call 4x8 micro-kernel with pre-packed B
            AddDot4x8_avx(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 8) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR; r++) {
                AddDot(k, a + (row + r) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows (< 4) with scalar
    CBLAS_INDEX remaining_rows = m - row;
    for (CBLAS_INDEX r = 0; r < remaining_rows; r++) {
        for (col = 0; col < n; col++) {
            AddDot(k, a + (row + r) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row + r), alpha);
        }
    }
    
    if (!use_pool_a) free(packedA);
    if (!use_pool_b) free(packedB);
}

//------------------------------------------------------
// SGEMM kernel - AVX 256-bit version (non-FMA)
//------------------------------------------------------
void sgemm_k_avx(cblas_args_t* args)
{
    InnerKernel_avx(args->ib, args->n, args->pb, 
                    args->a, args->lda, 
                    args->b, args->ldb, 
                    args->c, args->ldc,
                    args->alpha_s, args->thread_id,
                    args->transa, args->transb);
}

#endif // x86_64
