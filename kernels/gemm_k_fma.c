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
// 6x16 micro-kernel using AVX2+FMA
// Computes C[6x16] += alpha * A[6xk] * B[kx16]
// Uses 12 YMM registers for C (6 rows × 2 YMM per row)
// 2 YMM for B loads, 1 YMM for A broadcast
// K-loop unrolled by 4 for better ILP
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
    
    CBLAS_INDEX p = 0;
    
    // Main loop: unrolled by 4 for better ILP
    for (; p + 4 <= k; p += 4)
    {
        // Prefetch ahead
        CBLAS_PREFETCH(a + (4 * MR), 0, 3);
        CBLAS_PREFETCH(b + (4 * NR), 0, 3);
        
        // Iteration 0
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
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
        a_elem = _mm256_set1_ps(a[4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        a_elem = _mm256_set1_ps(a[5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        // Iteration 1
        b0 = _mm256_loadu_ps(b + NR);
        b1 = _mm256_loadu_ps(b + NR + 8);
        a_elem = _mm256_set1_ps(a[MR]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        a_elem = _mm256_set1_ps(a[MR + 1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        a_elem = _mm256_set1_ps(a[MR + 2]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        a_elem = _mm256_set1_ps(a[MR + 3]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        a_elem = _mm256_set1_ps(a[MR + 4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        a_elem = _mm256_set1_ps(a[MR + 5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        // Iteration 2
        b0 = _mm256_loadu_ps(b + 2*NR);
        b1 = _mm256_loadu_ps(b + 2*NR + 8);
        a_elem = _mm256_set1_ps(a[2*MR]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        a_elem = _mm256_set1_ps(a[2*MR + 1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        a_elem = _mm256_set1_ps(a[2*MR + 2]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        a_elem = _mm256_set1_ps(a[2*MR + 3]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        a_elem = _mm256_set1_ps(a[2*MR + 4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        a_elem = _mm256_set1_ps(a[2*MR + 5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        // Iteration 3
        b0 = _mm256_loadu_ps(b + 3*NR);
        b1 = _mm256_loadu_ps(b + 3*NR + 8);
        a_elem = _mm256_set1_ps(a[3*MR]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        a_elem = _mm256_set1_ps(a[3*MR + 1]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        a_elem = _mm256_set1_ps(a[3*MR + 2]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        a_elem = _mm256_set1_ps(a[3*MR + 3]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        a_elem = _mm256_set1_ps(a[3*MR + 4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        a_elem = _mm256_set1_ps(a[3*MR + 5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        b += 4 * NR;
        a += 4 * MR;
    }
    
    // Handle remaining k iterations (0-3)
    for (; p < k; p++)
    {
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
        
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
        a_elem = _mm256_set1_ps(a[4]);
        c40 = _mm256_fmadd_ps(a_elem, b0, c40);
        c41 = _mm256_fmadd_ps(a_elem, b1, c41);
        a_elem = _mm256_set1_ps(a[5]);
        c50 = _mm256_fmadd_ps(a_elem, b0, c50);
        c51 = _mm256_fmadd_ps(a_elem, b1, c51);
        
        b += NR;
        a += MR;
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
// K-loop unrolled by 2 for better ILP
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
    
    CBLAS_INDEX p = 0;
    
    // Unrolled by 2
    for (; p + 2 <= k; p += 2)
    {
        // Iteration 0
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
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
        
        // Iteration 1
        b0 = _mm256_loadu_ps(b + NR);
        b1 = _mm256_loadu_ps(b + NR + 8);
        a_elem = _mm256_set1_ps(a[4]);
        c00 = _mm256_fmadd_ps(a_elem, b0, c00);
        c01 = _mm256_fmadd_ps(a_elem, b1, c01);
        a_elem = _mm256_set1_ps(a[5]);
        c10 = _mm256_fmadd_ps(a_elem, b0, c10);
        c11 = _mm256_fmadd_ps(a_elem, b1, c11);
        a_elem = _mm256_set1_ps(a[6]);
        c20 = _mm256_fmadd_ps(a_elem, b0, c20);
        c21 = _mm256_fmadd_ps(a_elem, b1, c21);
        a_elem = _mm256_set1_ps(a[7]);
        c30 = _mm256_fmadd_ps(a_elem, b0, c30);
        c31 = _mm256_fmadd_ps(a_elem, b1, c31);
        
        b += 2 * NR;
        a += 2 * 4;
    }
    
    // Handle remaining iteration
    for (; p < k; p++)
    {
        b0 = _mm256_loadu_ps(b);
        b1 = _mm256_loadu_ps(b + 8);
        
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
        
        b += NR;
        a += 4;
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
// PackMatrixA_6_trans - Pack 6×k panel of A from transposed storage
// Transposed: logical A[row, col] = a[col*lda + row]
//------------------------------------------------------
static void PackMatrixA_6_trans(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *a_col = a + i * lda;

        if (i + 8 < k) {
            CBLAS_PREFETCH(a + (i + 8) * lda, 0, 3);
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
// PackMatrixA_4_wide_trans - Pack 4 rows from transposed storage
//------------------------------------------------------
static void PackMatrixA_4_wide_trans(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *a_col = a + i * lda;
        a_to[0] = a_col[0];
        a_to[1] = a_col[1];
        a_to[2] = a_col[2];
        a_to[3] = a_col[3];
        a_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixA_2_trans - Pack 2 rows from transposed storage
//------------------------------------------------------
static void PackMatrixA_2_trans(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *a_col = a + i * lda;
        a_to[0] = a_col[0];
        a_to[1] = a_col[1];
        a_to += 2;
    }
}

//------------------------------------------------------
// PackMatrixA_1_trans - Pack 1 row from transposed storage
//------------------------------------------------------
static void PackMatrixA_1_trans(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[i] = a[i * lda];
    }
}

//------------------------------------------------------
// PackMatrixB_16_trans - Pack k×16 panel of B from transposed storage
// Transposed: logical B[row, col] = b[col*ldb + row]
// For row p, gather b[0*ldb+p], b[1*ldb+p], ..., b[15*ldb+p]
//------------------------------------------------------
static void PackMatrixB_16_trans(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        CBLAS_INDEX col;
        for (col = 0; col < n_cols; col++) {
            b_to[col] = b[col * ldb + j];
        }
        for (; col < NR; col++) {
            b_to[col] = 0.0f;
        }
        b_to += NR;
    }
}

//------------------------------------------------------
// InnerKernel - FMA implementation with 6x16 micro-kernel
// Pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_fma(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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

    // Transpose-aware strides for A
    CBLAS_INDEX a_row_stride = transA ? 1 : lda;
    CBLAS_INDEX a_col_stride = transA ? lda : 1;

    // Transpose-aware strides for B
    CBLAS_INDEX b_row_stride = transB ? 1 : ldb;
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    CBLAS_INDEX row, col;

    // Phase 1: Pack ALL B column panels once upfront
    for (col = 0; col + NR <= n; col += NR)
    {
        float *b_ptr = b + col * b_col_stride;
        if (transB)
            PackMatrixB_16_trans(k, NR, b_ptr, ldb, &packedB[col * k]);
        else
            PackMatrixB_16(k, NR, b_ptr, ldb, &packedB[col * k]);
    }

    // Phase 2: Iterate rows using pre-packed B

    // Main loop: 6 rows at a time
    for (row = 0; row + MR <= m; row += MR)
    {
        float *a_ptr = a + row * a_row_stride;
        if (transA)
            PackMatrixA_6_trans(k, MR, a_ptr, lda, packedA);
        else
            PackMatrixA_6(k, MR, a_ptr, lda, packedA);

        for (col = 0; col + NR <= n; col += NR)
        {
            AddDot6x16_fma(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 16) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR; r++) {
                AddDot(k, a + (row + r) * a_row_stride, a_col_stride,
                       b + col * b_col_stride, b_row_stride, &C(col, row + r), alpha);
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
            float *a_ptr = a + row * a_row_stride;
            if (transA)
                PackMatrixA_4_wide_trans(k, a_ptr, lda, packedA);
            else
                PackMatrixA_4_wide(k, a_ptr, lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                AddDot4x16_fma(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
            }
            
            for (; col < n; col++) {
                for (CBLAS_INDEX r = 0; r < 4; r++) {
                    AddDot(k, a + (row + r) * a_row_stride, a_col_stride,
                           b + col * b_col_stride, b_row_stride, &C(col, row + r), alpha);
                }
            }
            row += 4;
            remaining_rows -= 4;
        }
        
        // Use 2x16 kernel for 2-3 remaining rows
        if (remaining_rows >= 2)
        {
            float *a_ptr = a + row * a_row_stride;
            if (transA)
                PackMatrixA_2_trans(k, a_ptr, lda, packedA);
            else
                PackMatrixA_2(k, a_ptr, lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                AddDot2x16_fma(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
            }
            
            for (; col < n; col++) {
                AddDot(k, a + row * a_row_stride, a_col_stride,
                       b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
                AddDot(k, a + (row + 1) * a_row_stride, a_col_stride,
                       b + col * b_col_stride, b_row_stride, &C(col, row + 1), alpha);
            }
            row += 2;
            remaining_rows -= 2;
        }
        
        // Use 1x16 kernel for last remaining row
        if (remaining_rows == 1)
        {
            float *a_ptr = a + row * a_row_stride;
            if (transA)
                PackMatrixA_1_trans(k, a_ptr, lda, packedA);
            else
                PackMatrixA_1(k, a_ptr, lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                AddDot1x16_fma(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
            }
            
            for (; col < n; col++) {
                AddDot(k, a + row * a_row_stride, a_col_stride,
                       b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
            }
        }
    }
    
    if (!use_pool_a) free(packedA);
    if (!use_pool_b) free(packedB);
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
                    args->alpha_s, args->thread_id,
                    args->transa, args->transb);
}

#endif // x86_64
