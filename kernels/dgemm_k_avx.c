//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// DGEMM kernel - AVX implementation for x86-64 (256-bit, non-FMA)
// Implements 4x4 micro-kernel using __m256d (4 doubles per register)
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for AVX 256-bit double
// 4x4 = 4 YMM registers for C (4 rows × 1 YMM of 4 doubles)
#define MR_D 4   // Rows per micro-kernel
#define NR_D 4   // Columns per micro-kernel (1 YMM register of 4 doubles)

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product (scalar fallback)
//------------------------------------------------------
static void AddDot_d(CBLAS_INDEX k, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *gamma, double alpha)
{
    double sum = 0.0;
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        sum += x[p * incx] * y[p * incy];
    }
    *gamma += alpha * sum;
}

//------------------------------------------------------
// 4x4 micro-kernel using AVX 256-bit for double precision (non-FMA)
// Computes C[4x4] += alpha * A[4xk] * B[kx4]
//------------------------------------------------------
static void AddDot4x4_dgemm_avx(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    // C accumulator registers: 4 rows × 1 YMM = 4 columns
    __m256d c0, c1, c2, c3;  // Rows 0-3, columns 0-3
    
    __m256d b_row;    // B row: 4 doubles = 1 YMM
    __m256d a_elem;   // A element broadcast
    
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    // Initialize accumulators to zero
    c0 = _mm256_setzero_pd();
    c1 = _mm256_setzero_pd();
    c2 = _mm256_setzero_pd();
    c3 = _mm256_setzero_pd();
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (4 doubles from packed format)
        b_row = _mm256_loadu_pd(b);
        b += NR_D;
        
        // Prefetch
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR_D), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR_D), 0, 3);
        }
        
        // Row 0: broadcast A[0,p] and multiply-add (non-FMA)
        a_elem = _mm256_set1_pd(a[0]);
        c0 = _mm256_add_pd(c0, _mm256_mul_pd(a_elem, b_row));
        
        // Row 1
        a_elem = _mm256_set1_pd(a[1]);
        c1 = _mm256_add_pd(c1, _mm256_mul_pd(a_elem, b_row));
        
        // Row 2
        a_elem = _mm256_set1_pd(a[2]);
        c2 = _mm256_add_pd(c2, _mm256_mul_pd(a_elem, b_row));
        
        // Row 3
        a_elem = _mm256_set1_pd(a[3]);
        c3 = _mm256_add_pd(c3, _mm256_mul_pd(a_elem, b_row));
        
        a += MR_D;
    }
    
    // Load old C, apply alpha, accumulate, store
    __m256d c_old;
    
    c_old = _mm256_loadu_pd(&C(0, 0));
    c0 = _mm256_add_pd(c_old, _mm256_mul_pd(alpha_vec, c0));
    _mm256_storeu_pd(&C(0, 0), c0);
    
    c_old = _mm256_loadu_pd(&C(0, 1));
    c1 = _mm256_add_pd(c_old, _mm256_mul_pd(alpha_vec, c1));
    _mm256_storeu_pd(&C(0, 1), c1);
    
    c_old = _mm256_loadu_pd(&C(0, 2));
    c2 = _mm256_add_pd(c_old, _mm256_mul_pd(alpha_vec, c2));
    _mm256_storeu_pd(&C(0, 2), c2);
    
    c_old = _mm256_loadu_pd(&C(0, 3));
    c3 = _mm256_add_pd(c_old, _mm256_mul_pd(alpha_vec, c3));
    _mm256_storeu_pd(&C(0, 3), c3);
}

//------------------------------------------------------
// PackMatrixB_4_d - Copy a k×4 panel of B
//------------------------------------------------------
static void PackMatrixB_4_d(CBLAS_INDEX k, CBLAS_INDEX n_cols, double *b, CBLAS_INDEX ldb, double *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        double *b_ij_pntr = &B(0, j);
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR_D; col++) {
            b_to[col] = b_ij_pntr[col];
        }
        for (; col < NR_D; col++) {
            b_to[col] = 0.0;
        }
        b_to += NR_D;
    }
}

//------------------------------------------------------
// PackMatrixB_4_d_trans - Pack k×4 panel of B from transposed storage
// When B is transposed, logical B[p, col+c] = b[c*ldb + p]
//------------------------------------------------------
static void PackMatrixB_4_d_trans(CBLAS_INDEX k, CBLAS_INDEX n_cols, double *b, CBLAS_INDEX ldb, double *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR_D; col++) {
            b_to[col] = b[col * ldb + j];
        }
        for (; col < NR_D; col++) {
            b_to[col] = 0.0;
        }
        b_to += NR_D;
    }
}

//------------------------------------------------------
// PackMatrixA_4_d - Copy a 4×k panel of A
//------------------------------------------------------
static void PackMatrixA_4_d(CBLAS_INDEX k, CBLAS_INDEX m_rows, double *a, CBLAS_INDEX lda, double *a_to)
{
    double *a_ptrs[4];
    for (CBLAS_INDEX r = 0; r < MR_D; r++) {
        a_ptrs[r] = (r < m_rows) ? &A(0, r) : NULL;
    }
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        for (CBLAS_INDEX r = 0; r < MR_D; r++) {
            a_to[r] = a_ptrs[r] ? *a_ptrs[r]++ : 0.0;
        }
        a_to += MR_D;
    }
}

//------------------------------------------------------
// PackMatrixA_4_d_trans - Pack 4×k panel of A from transposed storage
// When A is transposed, logical A[row+r, p] = a[p*lda + r]
//------------------------------------------------------
static void PackMatrixA_4_d_trans(CBLAS_INDEX k, CBLAS_INDEX m_rows, double *a, CBLAS_INDEX lda, double *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        double *a_col = a + i * lda;
        
        if (i + 8 < k) {
            CBLAS_PREFETCH(a + (i + 8) * lda, 0, 3);
        }
        
        for (CBLAS_INDEX r = 0; r < MR_D; r++) {
            if (r < m_rows) {
                a_to[r] = a_col[r];
            } else {
                a_to[r] = 0.0;
            }
        }
        
        a_to += MR_D;
    }
}

//------------------------------------------------------
// InnerKernel - AVX implementation with 4x4 micro-kernel
//------------------------------------------------------
static void InnerKernel_dgemm_avx(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                                  double* a, CBLAS_INDEX lda, 
                                  double* b, CBLAS_INDEX ldb, 
                                  double* c, CBLAS_INDEX ldc,
                                  double alpha, int thread_id,
                                  CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb)
{
    cblas_gemm_buffer_t* buf = cblas_get_gemm_buffer(thread_id);
    double* packedA;
    double* packedB;
    int use_pool_a = 0, use_pool_b = 0;
    size_t packedB_needed = (size_t)k * n * sizeof(double);
    size_t pool_b_size = (size_t)cblas_gemm_kc * cblas_gemm_nb * sizeof(double);
    
    if (buf) {
        packedA = buf->packedA_d;
        use_pool_a = 1;
        if (packedB_needed <= pool_b_size) {
            packedB = buf->packedB_d;
            use_pool_b = 1;
        } else {
            packedB = (double*)malloc(packedB_needed);
            if (!packedB) return;
        }
    } else {
        packedA = (double*)malloc(MR_D * k * sizeof(double));
        packedB = (double*)malloc(packedB_needed);
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

    // Phase 1: Pack all k×NR_D panels of B upfront
    for (col = 0; col + NR_D <= n; col += NR_D)
    {
        double *b_ptr = b + col * b_col_stride;
        if (transB)
            PackMatrixB_4_d_trans(k, NR_D, b_ptr, ldb, &packedB[col * k]);
        else
            PackMatrixB_4_d(k, NR_D, b_ptr, ldb, &packedB[col * k]);
    }

    for (row = 0; row + MR_D <= m; row += MR_D)
    {
        if (transA)
            PackMatrixA_4_d_trans(k, MR_D, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_4_d(k, MR_D, a + row * a_row_stride, lda, packedA);

        for (col = 0; col + NR_D <= n; col += NR_D)
        {
            AddDot4x4_dgemm_avx(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
        }

        // Leftover columns
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR_D; r++) {
                AddDot_d(k, a + (row + r) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row + r), alpha);
            }
        }
    }

    // Leftover rows
    for (; row < m; row++) {
        for (col = 0; col < n; col++) {
            AddDot_d(k, a + (row) * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
        }
    }
    
    if (!use_pool_a) free(packedA);
    if (!use_pool_b) free(packedB);
}

//------------------------------------------------------
// DGEMM kernel - AVX version (256-bit, non-FMA)
//------------------------------------------------------
void dgemm_k_avx(cblas_args_t* args)
{
    InnerKernel_dgemm_avx(args->ib, args->n, args->pb, 
                          (double*)args->a, args->lda, 
                          (double*)args->b, args->ldb, 
                          (double*)args->c, args->ldc,
                          args->alpha_d, args->thread_id,
                          args->transa, args->transb);
}

#endif // x86_64
