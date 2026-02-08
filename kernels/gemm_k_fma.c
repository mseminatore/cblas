//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - FMA implementation for x86-64
// Implements 6x16 micro-kernel for optimal AVX2+FMA register utilization
//------------------------------------------------------

#include "cblas.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Micro-kernel dimensions
#define MR 6   // Rows per micro-kernel
#define NR 16  // Columns per micro-kernel (2 YMM registers of 8 floats)

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product of row of X and col of Y (scalar fallback)
//------------------------------------------------------
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
// compute 16 dot products at a time, 4 cols x 4 rows (FMA)
// alpha is applied when storing results back to C
//------------------------------------------------------
static void AddDot4x4_fma(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc, float alpha)
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

        // rows 1 - 4 using FMA
        c_row1 = _mm_fmadd_ps(a_p0, b_row, c_row1);
        c_row2 = _mm_fmadd_ps(a_p1, b_row, c_row2);
        c_row3 = _mm_fmadd_ps(a_p2, b_row, c_row3);
        c_row4 = _mm_fmadd_ps(a_p3, b_row, c_row4);
    }

    // Apply alpha and accumulate: C = C + alpha * (A*B)
    c_row1 = _mm_fmadd_ps(alpha_vec, c_row1, c_old1);
    c_row2 = _mm_fmadd_ps(alpha_vec, c_row2, c_old2);
    c_row3 = _mm_fmadd_ps(alpha_vec, c_row3, c_old3);
    c_row4 = _mm_fmadd_ps(alpha_vec, c_row4, c_old4);

    // Store results
    _mm_storeu_ps(&C(0, 0), c_row1);
    _mm_storeu_ps(&C(0, 1), c_row2);
    _mm_storeu_ps(&C(0, 2), c_row3);
    _mm_storeu_ps(&C(0, 3), c_row4);
}

//------------------------------------------------------
// 6x16 micro-kernel using AVX2+FMA
// Computes C[6x16] += alpha * A[6xk] * B[kx16]
// Uses 12 YMM registers for C (6 rows × 2 YMM per row)
// 2 YMM for B loads, 1 YMM for A broadcast
//------------------------------------------------------
static void AddDot6x16_fma(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 6 rows × 2 YMM (8 floats each) = 16 columns
    __m256 c00, c01;  // Row 0: columns 0-7, 8-15
    __m256 c10, c11;  // Row 1
    __m256 c20, c21;  // Row 2
    __m256 c30, c31;  // Row 3
    __m256 c40, c41;  // Row 4
    __m256 c50, c51;  // Row 5
    
    __m256 b0, b1;    // B row: 16 floats = 2 YMM
    __m256 a_elem;    // A element broadcast
    
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    // Initialize accumulators to zero
    c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
    c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
    c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
    c40 = _mm256_setzero_ps(); c41 = _mm256_setzero_ps();
    c50 = _mm256_setzero_ps(); c51 = _mm256_setzero_ps();
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (16 floats from packed format: 16 consecutive floats)
        b0 = _mm256_loadu_ps(b);      // B[p, 0:7]
        b1 = _mm256_loadu_ps(b + 8);  // B[p, 8:15]
        b += NR;  // Advance to next row of packed B
        
        // Prefetch next iterations
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR), 0, 3);
        }
        
        // Row 0: broadcast A[0,p] and FMA
        a_elem = _mm256_set1_ps(a[0]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        
        // Row 1
        a_elem = _mm256_set1_ps(a[1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        
        // Row 2
        a_elem = _mm256_set1_ps(a[2]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        
        // Row 3
        a_elem = _mm256_set1_ps(a[3]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        
        // Row 4
        a_elem = _mm256_set1_ps(a[4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        
        // Row 5
        a_elem = _mm256_set1_ps(a[5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        a += MR;  // Advance to next column of packed A
    }
    
    // Load old C values, apply alpha to accumulators, accumulate, store
    // Row 0
    __m256 c_old0 = _mm256_loadu_ps(&C(0, 0));
    __m256 c_old1 = _mm256_loadu_ps(&C(8, 0));
    c00 = _mm256_fmadd_ps(alpha_vec, c00, c_old0);
    c01 = _mm256_fmadd_ps(alpha_vec, c01, c_old1);
    _mm256_storeu_ps(&C(0, 0), c00);
    _mm256_storeu_ps(&C(8, 0), c01);
    
    // Row 1
    c_old0 = _mm256_loadu_ps(&C(0, 1));
    c_old1 = _mm256_loadu_ps(&C(8, 1));
    c10 = _mm256_fmadd_ps(alpha_vec, c10, c_old0);
    c11 = _mm256_fmadd_ps(alpha_vec, c11, c_old1);
    _mm256_storeu_ps(&C(0, 1), c10);
    _mm256_storeu_ps(&C(8, 1), c11);
    
    // Row 2
    c_old0 = _mm256_loadu_ps(&C(0, 2));
    c_old1 = _mm256_loadu_ps(&C(8, 2));
    c20 = _mm256_fmadd_ps(alpha_vec, c20, c_old0);
    c21 = _mm256_fmadd_ps(alpha_vec, c21, c_old1);
    _mm256_storeu_ps(&C(0, 2), c20);
    _mm256_storeu_ps(&C(8, 2), c21);
    
    // Row 3
    c_old0 = _mm256_loadu_ps(&C(0, 3));
    c_old1 = _mm256_loadu_ps(&C(8, 3));
    c30 = _mm256_fmadd_ps(alpha_vec, c30, c_old0);
    c31 = _mm256_fmadd_ps(alpha_vec, c31, c_old1);
    _mm256_storeu_ps(&C(0, 3), c30);
    _mm256_storeu_ps(&C(8, 3), c31);
    
    // Row 4
    c_old0 = _mm256_loadu_ps(&C(0, 4));
    c_old1 = _mm256_loadu_ps(&C(8, 4));
    c40 = _mm256_fmadd_ps(alpha_vec, c40, c_old0);
    c41 = _mm256_fmadd_ps(alpha_vec, c41, c_old1);
    _mm256_storeu_ps(&C(0, 4), c40);
    _mm256_storeu_ps(&C(8, 4), c41);
    
    // Row 5
    c_old0 = _mm256_loadu_ps(&C(0, 5));
    c_old1 = _mm256_loadu_ps(&C(8, 5));
    c50 = _mm256_fmadd_ps(alpha_vec, c50, c_old0);
    c51 = _mm256_fmadd_ps(alpha_vec, c51, c_old1);
    _mm256_storeu_ps(&C(0, 5), c50);
    _mm256_storeu_ps(&C(8, 5), c51);
}

//------------------------------------------------------
// 4x16 micro-kernel using AVX2+FMA
// For remainder handling when 4 <= remaining_rows < 6
//------------------------------------------------------
static void AddDot4x16_fma(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 4 rows × 2 YMM = 8 registers
    __m256 c00, c01;  // Row 0
    __m256 c10, c11;  // Row 1
    __m256 c20, c21;  // Row 2
    __m256 c30, c31;  // Row 3
    
    __m256 b0, b1;
    __m256 a_elem;
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
    c20 = _mm256_setzero_ps(); c21 = _mm256_setzero_ps();
    c30 = _mm256_setzero_ps(); c31 = _mm256_setzero_ps();
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
        b += NR;
        
        a_elem = _mm256_set1_ps(a[0]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        
        a_elem = _mm256_set1_ps(a[1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        
        a_elem = _mm256_set1_ps(a[2]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        
        a_elem = _mm256_set1_ps(a[3]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        
        a += 4;  // 4 rows packed
    }
    
    // Store results
    __m256 c_old0, c_old1;
    
    c_old0 = _mm256_loadu_ps(&C(0, 0));
    c_old1 = _mm256_loadu_ps(&C(8, 0));
    c00 = _mm256_fmadd_ps(alpha_vec, c00, c_old0);
    c01 = _mm256_fmadd_ps(alpha_vec, c01, c_old1);
    _mm256_storeu_ps(&C(0, 0), c00);
    _mm256_storeu_ps(&C(8, 0), c01);
    
    c_old0 = _mm256_loadu_ps(&C(0, 1));
    c_old1 = _mm256_loadu_ps(&C(8, 1));
    c10 = _mm256_fmadd_ps(alpha_vec, c10, c_old0);
    c11 = _mm256_fmadd_ps(alpha_vec, c11, c_old1);
    _mm256_storeu_ps(&C(0, 1), c10);
    _mm256_storeu_ps(&C(8, 1), c11);
    
    c_old0 = _mm256_loadu_ps(&C(0, 2));
    c_old1 = _mm256_loadu_ps(&C(8, 2));
    c20 = _mm256_fmadd_ps(alpha_vec, c20, c_old0);
    c21 = _mm256_fmadd_ps(alpha_vec, c21, c_old1);
    _mm256_storeu_ps(&C(0, 2), c20);
    _mm256_storeu_ps(&C(8, 2), c21);
    
    c_old0 = _mm256_loadu_ps(&C(0, 3));
    c_old1 = _mm256_loadu_ps(&C(8, 3));
    c30 = _mm256_fmadd_ps(alpha_vec, c30, c_old0);
    c31 = _mm256_fmadd_ps(alpha_vec, c31, c_old1);
    _mm256_storeu_ps(&C(0, 3), c30);
    _mm256_storeu_ps(&C(8, 3), c31);
}

//------------------------------------------------------
// 2x16 micro-kernel using AVX2+FMA
// For remainder handling when 2 <= remaining_rows < 4
//------------------------------------------------------
static void AddDot2x16_fma(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    __m256 c00, c01;  // Row 0
    __m256 c10, c11;  // Row 1
    
    __m256 b0, b1;
    __m256 a_elem;
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    c00 = _mm256_setzero_ps(); c01 = _mm256_setzero_ps();
    c10 = _mm256_setzero_ps(); c11 = _mm256_setzero_ps();
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
        b += NR;
        
        a_elem = _mm256_set1_ps(a[0]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        
        a_elem = _mm256_set1_ps(a[1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        
        a += 2;  // 2 rows packed
    }
    
    // Store results
    __m256 c_old0, c_old1;
    
    c_old0 = _mm256_loadu_ps(&C(0, 0));
    c_old1 = _mm256_loadu_ps(&C(8, 0));
    c00 = _mm256_fmadd_ps(alpha_vec, c00, c_old0);
    c01 = _mm256_fmadd_ps(alpha_vec, c01, c_old1);
    _mm256_storeu_ps(&C(0, 0), c00);
    _mm256_storeu_ps(&C(8, 0), c01);
    
    c_old0 = _mm256_loadu_ps(&C(0, 1));
    c_old1 = _mm256_loadu_ps(&C(8, 1));
    c10 = _mm256_fmadd_ps(alpha_vec, c10, c_old0);
    c11 = _mm256_fmadd_ps(alpha_vec, c11, c_old1);
    _mm256_storeu_ps(&C(0, 1), c10);
    _mm256_storeu_ps(&C(8, 1), c11);
}

//------------------------------------------------------
// 1x16 micro-kernel using AVX2+FMA
// For remainder handling when remaining_rows == 1
//------------------------------------------------------
static void AddDot1x16_fma(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    (void)ldc;  // Not used for single row
    
    __m256 c00, c01;  // Row 0
    __m256 b0, b1;
    __m256 a_elem;
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    
    c00 = _mm256_setzero_ps();
    c01 = _mm256_setzero_ps();
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
        b += NR;
        
        a_elem = _mm256_set1_ps(a[0]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        
        a += 1;
    }
    
    // Store results
    __m256 c_old0 = _mm256_loadu_ps(&C(0, 0));
    __m256 c_old1 = _mm256_loadu_ps(&C(8, 0));
    c00 = _mm256_fmadd_ps(alpha_vec, c00, c_old0);
    c01 = _mm256_fmadd_ps(alpha_vec, c01, c_old1);
    _mm256_storeu_ps(&C(0, 0), c00);
    _mm256_storeu_ps(&C(8, 0), c01);
}

//------------------------------------------------------
// PackMatrixA_4_wide - Pack 4 rows for use with 4x16 kernel
//------------------------------------------------------
static void PackMatrixA_4_wide(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_ptrs[4] = { &A(0,0), &A(0,1), &A(0,2), &A(0,3) };
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[0] = *a_ptrs[0]++;
        a_to[1] = *a_ptrs[1]++;
        a_to[2] = *a_ptrs[2]++;
        a_to[3] = *a_ptrs[3]++;
        a_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixA_2 - Pack 2 rows for use with 2x16 kernel
//------------------------------------------------------
static void PackMatrixA_2(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_0 = &A(0,0);
    float *a_1 = &A(0,1);
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[0] = *a_0++;
        a_to[1] = *a_1++;
        a_to += 2;
    }
}

//------------------------------------------------------
// PackMatrixA_1 - Pack 1 row for use with 1x16 kernel
//------------------------------------------------------
static void PackMatrixA_1(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_0 = &A(0,0);
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[i] = *a_0++;
    }
}

//------------------------------------------------------
// PackMatrixB - Copy a k×4 panel of B into contiguous memory (for 4x4 kernel)
//------------------------------------------------------
static void PackMatrixB_4(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
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
// PackMatrixA - Copy a 4×k panel of A into contiguous memory (for 4x4 kernel)
//------------------------------------------------------
static void PackMatrixA_4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
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
// PackMatrixB_16 - Copy a k×16 panel of B into contiguous memory
// Packing format: For each row p of B, store 16 consecutive columns
// Vectorized version using AVX for full-width copies
//------------------------------------------------------
static void PackMatrixB_16(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    if (n_cols >= NR) {
        // Fast path: full 16 columns, use AVX loads/stores
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            
            // Prefetch ahead
            if (j + 4 < k) {
                CBLAS_PREFETCH(&B(0, j + 4), 0, 3);
            }
            
            // Load and store 16 floats using AVX (2 YMM registers)
            __m256 b0 = _mm256_loadu_ps(b_ij_pntr);
            __m256 b1 = _mm256_loadu_ps(b_ij_pntr + 8);
            _mm256_storeu_ps(b_to, b0);
            _mm256_storeu_ps(b_to + 8, b1);
            
            b_to += NR;
        }
    } else {
        // Slow path: partial columns, need zero-padding
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            
            if (j + 4 < k) {
                CBLAS_PREFETCH(&B(0, j + 4), 0, 3);
            }
            
            // Copy available columns
            CBLAS_INDEX col;
            for (col = 0; col < n_cols; col++) {
                b_to[col] = b_ij_pntr[col];
            }
            // Zero-pad remaining columns
            for (; col < NR; col++) {
                b_to[col] = 0.0f;
            }
            
            b_to += NR;
        }
    }
}

//------------------------------------------------------
// PackMatrixA_6 - Copy a 6×k panel of A into contiguous memory
// Packing format: For each column p of A, store 6 consecutive rows
//------------------------------------------------------
static void PackMatrixA_6(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    // Handle varying number of rows (1-6)
    float *a_ptrs[6];
    
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
            CBLAS_PREFETCH(a_ptrs[0] + 8, 0, 3);
        }
        
        // Pack 6 rows for this column
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
// InnerKernel - FMA implementation with 6x16 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
// InnerKernel - optimized inner kernel with 6x16 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_fma(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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
        packedA = (float*)malloc(MR * k * sizeof(float));
        packedB = (float*)malloc(k * NR * sizeof(float));
        if (!packedA || !packedB) {
            free(packedA);
            free(packedB);
            return;
        }
    }

    CBLAS_INDEX row, col;

    // Main loop: 6 rows at a time
    for (row = 0; row + MR <= m; row += MR)
    {
        // Pack this 6×k panel of A once per row iteration
        PackMatrixA_6(k, MR, &A(0, row), lda, packedA);

        // Process 16 columns at a time
        for (col = 0; col + NR <= n; col += NR)
        {
            // Pack this k×16 panel of B
            PackMatrixB_16(k, NR, &B(col, 0), ldb, packedB);
            
            // Call 6x16 micro-kernel
            AddDot6x16_fma(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 16) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR; r++) {
                AddDot(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows (< 6) using optimized remainder kernels
    CBLAS_INDEX remaining_rows = m - row;
    if (remaining_rows > 0)
    {
        // Use 4x16 kernel for 4-5 remaining rows
        if (remaining_rows >= 4)
        {
            PackMatrixA_4_wide(k, &A(0, row), lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                PackMatrixB_16(k, NR, &B(col, 0), ldb, packedB);
                AddDot4x16_fma(k, packedA, packedB, &C(col, row), ldc, alpha);
            }
            
            // Leftover columns with 4 rows - use scalar
            for (; col < n; col++) {
                for (CBLAS_INDEX r = 0; r < 4; r++) {
                    AddDot(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
                }
            }
            row += 4;
            remaining_rows -= 4;
        }
        
        // Use 2x16 kernel for 2-3 remaining rows
        if (remaining_rows >= 2)
        {
            PackMatrixA_2(k, &A(0, row), lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                PackMatrixB_16(k, NR, &B(col, 0), ldb, packedB);
                AddDot2x16_fma(k, packedA, packedB, &C(col, row), ldc, alpha);
            }
            
            // Leftover columns
            for (; col < n; col++) {
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
                AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1), alpha);
            }
            row += 2;
            remaining_rows -= 2;
        }
        
        // Use 1x16 kernel for last remaining row
        if (remaining_rows == 1)
        {
            PackMatrixA_1(k, &A(0, row), lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                PackMatrixB_16(k, NR, &B(col, 0), ldb, packedB);
                AddDot1x16_fma(k, packedA, packedB, &C(col, row), ldc, alpha);
            }
            
            // Leftover columns
            for (; col < n; col++) {
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
            }
        }
    }
    
    if (!use_pool) {
        free(packedA);
        free(packedB);
    }
}

//------------------------------------------------------
// SGEMM kernel - FMA version
//------------------------------------------------------
void sgemm_k_fma(cblas_args_t* args)
{
    InnerKernel_fma(args->ib, args->n, args->pb, 
                    args->a, args->lda, 
                    args->b, args->ldb, 
                    args->c, args->ldc,
                    args->alpha_s, args->thread_id);
}

#endif // x86_64
