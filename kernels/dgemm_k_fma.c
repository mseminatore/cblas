//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// DGEMM kernel - FMA implementation for x86-64
// Implements 6x8 micro-kernel for AVX2+FMA with double precision
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for double precision
// 6x8 = 12 YMM registers for C (6 rows × 2 YMM of 4 doubles each)
#define MR_D 6   // Rows per micro-kernel
#define NR_D 8   // Columns per micro-kernel (2 YMM registers of 4 doubles)

// Smaller kernel for cleanup
#define MR_D_SMALL 4
#define NR_D_SMALL 4

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product of row of X and col of Y (scalar fallback)
// alpha is applied to the result before accumulating
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
// 6x8 micro-kernel using AVX2+FMA for double precision
// Computes C[6x8] += alpha * A[6xk] * B[kx8]
// Uses 12 YMM registers for C (6 rows × 2 YMM per row)
// 2 YMM for B loads, 1 for A broadcast
//------------------------------------------------------
static void AddDot6x8_dgemm_fma(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    // C accumulator registers: 6 rows × 2 YMM = 8 columns
    __m256d c00, c01;  // Row 0: columns 0-3, 4-7
    __m256d c10, c11;  // Row 1
    __m256d c20, c21;  // Row 2
    __m256d c30, c31;  // Row 3
    __m256d c40, c41;  // Row 4
    __m256d c50, c51;  // Row 5
    
    __m256d b0, b1;    // B row: 8 doubles = 2 YMM
    __m256d a_elem;    // A element broadcast
    
    __m256d alpha_vec = _mm256_set1_pd(alpha);
    
    // Initialize accumulators to zero
    c00 = _mm256_setzero_pd(); c01 = _mm256_setzero_pd();
    c10 = _mm256_setzero_pd(); c11 = _mm256_setzero_pd();
    c20 = _mm256_setzero_pd(); c21 = _mm256_setzero_pd();
    c30 = _mm256_setzero_pd(); c31 = _mm256_setzero_pd();
    c40 = _mm256_setzero_pd(); c41 = _mm256_setzero_pd();
    c50 = _mm256_setzero_pd(); c51 = _mm256_setzero_pd();
    
    // Main loop over k dimension with 2x unrolling
    CBLAS_INDEX p = 0;
    for (; p + 1 < k; p += 2)
    {
        // === Iteration 0 ===
        // Load B row (8 doubles from packed format)
        b0 = _mm256_loadu_pd(b);      // B[p, 0:3]
        b1 = _mm256_loadu_pd(b + 4);  // B[p, 4:7]
        
        // Prefetch next iterations
        CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR_D), 0, 3);
        CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR_D), 0, 3);
        
        // Row 0
        a_elem = _mm256_broadcast_sd(&a[0]);
        c00 = _mm256_fmadd_pd(a_elem, b0, c00);
        c01 = _mm256_fmadd_pd(a_elem, b1, c01);
        
        // Row 1
        a_elem = _mm256_broadcast_sd(&a[1]);
        c10 = _mm256_fmadd_pd(a_elem, b0, c10);
        c11 = _mm256_fmadd_pd(a_elem, b1, c11);
        
        // Row 2
        a_elem = _mm256_broadcast_sd(&a[2]);
        c20 = _mm256_fmadd_pd(a_elem, b0, c20);
        c21 = _mm256_fmadd_pd(a_elem, b1, c21);
        
        // Row 3
        a_elem = _mm256_broadcast_sd(&a[3]);
        c30 = _mm256_fmadd_pd(a_elem, b0, c30);
        c31 = _mm256_fmadd_pd(a_elem, b1, c31);
        
        // Row 4
        a_elem = _mm256_broadcast_sd(&a[4]);
        c40 = _mm256_fmadd_pd(a_elem, b0, c40);
        c41 = _mm256_fmadd_pd(a_elem, b1, c41);
        
        // Row 5
        a_elem = _mm256_broadcast_sd(&a[5]);
        c50 = _mm256_fmadd_pd(a_elem, b0, c50);
        c51 = _mm256_fmadd_pd(a_elem, b1, c51);
        
        a += MR_D;
        b += NR_D;
        
        // === Iteration 1 ===
        b0 = _mm256_loadu_pd(b);
        b1 = _mm256_loadu_pd(b + 4);
        
        a_elem = _mm256_broadcast_sd(&a[0]);
        c00 = _mm256_fmadd_pd(a_elem, b0, c00);
        c01 = _mm256_fmadd_pd(a_elem, b1, c01);
        
        a_elem = _mm256_broadcast_sd(&a[1]);
        c10 = _mm256_fmadd_pd(a_elem, b0, c10);
        c11 = _mm256_fmadd_pd(a_elem, b1, c11);
        
        a_elem = _mm256_broadcast_sd(&a[2]);
        c20 = _mm256_fmadd_pd(a_elem, b0, c20);
        c21 = _mm256_fmadd_pd(a_elem, b1, c21);
        
        a_elem = _mm256_broadcast_sd(&a[3]);
        c30 = _mm256_fmadd_pd(a_elem, b0, c30);
        c31 = _mm256_fmadd_pd(a_elem, b1, c31);
        
        a_elem = _mm256_broadcast_sd(&a[4]);
        c40 = _mm256_fmadd_pd(a_elem, b0, c40);
        c41 = _mm256_fmadd_pd(a_elem, b1, c41);
        
        a_elem = _mm256_broadcast_sd(&a[5]);
        c50 = _mm256_fmadd_pd(a_elem, b0, c50);
        c51 = _mm256_fmadd_pd(a_elem, b1, c51);
        
        a += MR_D;
        b += NR_D;
    }
    
    // Handle odd k
    if (p < k)
    {
        b0 = _mm256_loadu_pd(b);
        b1 = _mm256_loadu_pd(b + 4);
        
        a_elem = _mm256_broadcast_sd(&a[0]);
        c00 = _mm256_fmadd_pd(a_elem, b0, c00);
        c01 = _mm256_fmadd_pd(a_elem, b1, c01);
        
        a_elem = _mm256_broadcast_sd(&a[1]);
        c10 = _mm256_fmadd_pd(a_elem, b0, c10);
        c11 = _mm256_fmadd_pd(a_elem, b1, c11);
        
        a_elem = _mm256_broadcast_sd(&a[2]);
        c20 = _mm256_fmadd_pd(a_elem, b0, c20);
        c21 = _mm256_fmadd_pd(a_elem, b1, c21);
        
        a_elem = _mm256_broadcast_sd(&a[3]);
        c30 = _mm256_fmadd_pd(a_elem, b0, c30);
        c31 = _mm256_fmadd_pd(a_elem, b1, c31);
        
        a_elem = _mm256_broadcast_sd(&a[4]);
        c40 = _mm256_fmadd_pd(a_elem, b0, c40);
        c41 = _mm256_fmadd_pd(a_elem, b1, c41);
        
        a_elem = _mm256_broadcast_sd(&a[5]);
        c50 = _mm256_fmadd_pd(a_elem, b0, c50);
        c51 = _mm256_fmadd_pd(a_elem, b1, c51);
    }
    
    // Load old C values, apply alpha to accumulators, accumulate, store
    __m256d c_old0, c_old1;
    
    // Row 0
    c_old0 = _mm256_loadu_pd(&C(0, 0));
    c_old1 = _mm256_loadu_pd(&C(4, 0));
    c00 = _mm256_fmadd_pd(alpha_vec, c00, c_old0);
    c01 = _mm256_fmadd_pd(alpha_vec, c01, c_old1);
    _mm256_storeu_pd(&C(0, 0), c00);
    _mm256_storeu_pd(&C(4, 0), c01);
    
    // Row 1
    c_old0 = _mm256_loadu_pd(&C(0, 1));
    c_old1 = _mm256_loadu_pd(&C(4, 1));
    c10 = _mm256_fmadd_pd(alpha_vec, c10, c_old0);
    c11 = _mm256_fmadd_pd(alpha_vec, c11, c_old1);
    _mm256_storeu_pd(&C(0, 1), c10);
    _mm256_storeu_pd(&C(4, 1), c11);
    
    // Row 2
    c_old0 = _mm256_loadu_pd(&C(0, 2));
    c_old1 = _mm256_loadu_pd(&C(4, 2));
    c20 = _mm256_fmadd_pd(alpha_vec, c20, c_old0);
    c21 = _mm256_fmadd_pd(alpha_vec, c21, c_old1);
    _mm256_storeu_pd(&C(0, 2), c20);
    _mm256_storeu_pd(&C(4, 2), c21);
    
    // Row 3
    c_old0 = _mm256_loadu_pd(&C(0, 3));
    c_old1 = _mm256_loadu_pd(&C(4, 3));
    c30 = _mm256_fmadd_pd(alpha_vec, c30, c_old0);
    c31 = _mm256_fmadd_pd(alpha_vec, c31, c_old1);
    _mm256_storeu_pd(&C(0, 3), c30);
    _mm256_storeu_pd(&C(4, 3), c31);
    
    // Row 4
    c_old0 = _mm256_loadu_pd(&C(0, 4));
    c_old1 = _mm256_loadu_pd(&C(4, 4));
    c40 = _mm256_fmadd_pd(alpha_vec, c40, c_old0);
    c41 = _mm256_fmadd_pd(alpha_vec, c41, c_old1);
    _mm256_storeu_pd(&C(0, 4), c40);
    _mm256_storeu_pd(&C(4, 4), c41);
    
    // Row 5
    c_old0 = _mm256_loadu_pd(&C(0, 5));
    c_old1 = _mm256_loadu_pd(&C(4, 5));
    c50 = _mm256_fmadd_pd(alpha_vec, c50, c_old0);
    c51 = _mm256_fmadd_pd(alpha_vec, c51, c_old1);
    _mm256_storeu_pd(&C(0, 5), c50);
    _mm256_storeu_pd(&C(4, 5), c51);
}

//------------------------------------------------------
// PackMatrixB_8_d - Copy a k×8 panel of B into contiguous memory
// Packing format: For each row p of B, store 8 consecutive columns
//------------------------------------------------------
static void PackMatrixB_8_d(CBLAS_INDEX k, CBLAS_INDEX n_cols, double *b, CBLAS_INDEX ldb, double *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        double *b_ij_pntr = &B(0, j);
        
        // Prefetch ahead
        if (j + 4 < k) {
            CBLAS_PREFETCH(&B(0, j + 4), 0, 3);
        }
        
        // Copy up to 8 columns, zero-pad if fewer
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR_D; col++) {
            b_to[col] = b_ij_pntr[col];
        }
        // Zero-pad remaining columns
        for (; col < NR_D; col++) {
            b_to[col] = 0.0;
        }
        
        b_to += NR_D;
    }
}

//------------------------------------------------------
// PackMatrixA_6_d - Copy a 6×k panel of A into contiguous memory
// Packing format: For each column p of A, store 6 consecutive rows
//------------------------------------------------------
static void PackMatrixA_6_d(CBLAS_INDEX k, CBLAS_INDEX m_rows, double *a, CBLAS_INDEX lda, double *a_to)
{
    // Handle varying number of rows (1-6)
    double *a_ptrs[6];
    
    for (CBLAS_INDEX r = 0; r < MR_D; r++) {
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
            CBLAS_PREFETCH(a_ptrs[0] + 8, 0, 3);
        }
        
        // Pack 6 rows for this column
        for (CBLAS_INDEX r = 0; r < MR_D; r++) {
            if (a_ptrs[r]) {
                a_to[r] = *a_ptrs[r]++;
            } else {
                a_to[r] = 0.0;  // Zero-pad
            }
        }
        
        a_to += MR_D;
    }
}

//------------------------------------------------------
// InnerKernel - FMA implementation with 6x8 micro-kernel for dgemm
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_dgemm_fma(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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

    // Main loop: 6 rows at a time
    for (row = 0; row + MR_D <= m; row += MR_D)
    {
        // Pack this 6×k panel of A once per row iteration
        PackMatrixA_6_d(k, MR_D, &A(0, row), lda, packedA);

        // Process 8 columns at a time
        for (col = 0; col + NR_D <= n; col += NR_D)
        {
            // Pack this k×8 panel of B
            PackMatrixB_8_d(k, NR_D, &B(col, 0), ldb, packedB);
            
            // Call 6x8 micro-kernel
            AddDot6x8_dgemm_fma(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 8) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR_D; r++) {
                AddDot_d(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows (< 6) with scalar
    CBLAS_INDEX remaining_rows = m - row;
    for (CBLAS_INDEX r = 0; r < remaining_rows; r++) {
        for (col = 0; col < n; col++) {
            AddDot_d(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
        }
    }
    
    if (!use_pool) {
        free(packedA);
        free(packedB);
    }
}

//------------------------------------------------------
// DGEMM kernel - FMA version
//------------------------------------------------------
void dgemm_k_fma(cblas_args_t* args)
{
    InnerKernel_dgemm_fma(args->ib, args->n, args->pb, 
                          (double*)args->a, args->lda, 
                          (double*)args->b, args->ldb, 
                          (double*)args->c, args->ldc,
                          args->alpha_d, args->thread_id);
}

#endif // x86_64
