//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// DGEMM kernel - SSE implementation for x86-64 (128-bit, non-FMA)
// Implements 2x2 micro-kernel using __m128d (2 doubles per register)
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for SSE 128-bit double
// 2x2 = 2 XMM registers for C (2 rows × 1 XMM of 2 doubles)
#define MR_D 2   // Rows per micro-kernel
#define NR_D 2   // Columns per micro-kernel (1 XMM register of 2 doubles)

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
// 2x2 micro-kernel using SSE 128-bit for double precision
// Computes C[2x2] += alpha * A[2xk] * B[kx2]
//------------------------------------------------------
static void AddDot2x2_dgemm_sse(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    // C accumulator registers: 2 rows × 1 XMM = 2 columns
    __m128d c0, c1;  // Rows 0-1, columns 0-1
    
    __m128d b_row;    // B row: 2 doubles = 1 XMM
    __m128d a_elem;   // A element broadcast
    
    __m128d alpha_vec = _mm_set1_pd(alpha);
    
    // Initialize accumulators to zero
    c0 = _mm_setzero_pd();
    c1 = _mm_setzero_pd();
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (2 doubles from packed format)
        b_row = _mm_loadu_pd(b);
        b += NR_D;
        
        // Prefetch
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR_D), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR_D), 0, 3);
        }
        
        // Row 0: broadcast A[0,p] and multiply-add
        a_elem = _mm_set1_pd(a[0]);
        c0 = _mm_add_pd(c0, _mm_mul_pd(a_elem, b_row));
        
        // Row 1
        a_elem = _mm_set1_pd(a[1]);
        c1 = _mm_add_pd(c1, _mm_mul_pd(a_elem, b_row));
        
        a += MR_D;
    }
    
    // Load old C, apply alpha, accumulate, store
    __m128d c_old;
    
    c_old = _mm_loadu_pd(&C(0, 0));
    c0 = _mm_add_pd(c_old, _mm_mul_pd(alpha_vec, c0));
    _mm_storeu_pd(&C(0, 0), c0);
    
    c_old = _mm_loadu_pd(&C(0, 1));
    c1 = _mm_add_pd(c_old, _mm_mul_pd(alpha_vec, c1));
    _mm_storeu_pd(&C(0, 1), c1);
}

//------------------------------------------------------
// PackMatrixB_2_d - Copy a k×2 panel of B
//------------------------------------------------------
static void PackMatrixB_2_d(CBLAS_INDEX k, CBLAS_INDEX n_cols, double *b, CBLAS_INDEX ldb, double *b_to)
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
// PackMatrixA_2_d - Copy a 2×k panel of A
//------------------------------------------------------
static void PackMatrixA_2_d(CBLAS_INDEX k, CBLAS_INDEX m_rows, double *a, CBLAS_INDEX lda, double *a_to)
{
    double *a_ptrs[2];
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
// InnerKernel - SSE implementation with 2x2 micro-kernel
//------------------------------------------------------
static void InnerKernel_dgemm_sse(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                                  double* a, CBLAS_INDEX lda, 
                                  double* b, CBLAS_INDEX ldb, 
                                  double* c, CBLAS_INDEX ldc,
                                  double alpha, int thread_id)
{
    // Temporarily disabled buffer pool - always use malloc
    double* packedA;
    double* packedB;
    int use_pool = 0;
    (void)thread_id;
    
    packedA = (double*)malloc(MR_D * k * sizeof(double));
    packedB = (double*)malloc(k * NR_D * sizeof(double));
    
    if (!packedA || !packedB) {
        free(packedA);
        free(packedB);
        return;
    }

    CBLAS_INDEX row, col;

    for (row = 0; row + MR_D <= m; row += MR_D)
    {
        PackMatrixA_2_d(k, MR_D, &A(0, row), lda, packedA);

        for (col = 0; col + NR_D <= n; col += NR_D)
        {
            PackMatrixB_2_d(k, NR_D, &B(col, 0), ldb, packedB);
            AddDot2x2_dgemm_sse(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Leftover columns
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR_D; r++) {
                AddDot_d(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Leftover rows
    for (; row < m; row++) {
        for (col = 0; col < n; col++) {
            AddDot_d(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
        }
    }
    
    if (!use_pool) {
        free(packedA);
        free(packedB);
    }
}

//------------------------------------------------------
// DGEMM kernel - SSE version (128-bit)
//------------------------------------------------------
void dgemm_k_sse(cblas_args_t* args)
{
    InnerKernel_dgemm_sse(args->ib, args->n, args->pb, 
                          (double*)args->a, args->lda, 
                          (double*)args->b, args->ldb, 
                          (double*)args->c, args->ldc,
                          args->alpha_d, args->thread_id);
}

#endif // x86_64
