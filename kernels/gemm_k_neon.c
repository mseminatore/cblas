//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - NEON implementation for ARM64
// Implements 8x12 micro-kernel for maximum register utilization
//------------------------------------------------------

#include "cblas.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for ARM64 NEON (8x12)
// 8x12 = 24 float32x4_t registers for C (8 rows × 3 registers of 4 floats)
// Uses 75% of ARM64's 32 SIMD registers
#define MR 8   // Rows per micro-kernel
#define NR 12  // Columns per micro-kernel (3 float32x4_t registers)

// Legacy dimensions for remainder handling
#define MR_LEGACY 4
#define NR_LEGACY 8

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
    float sum = 0.0f;

    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        sum += x[p * incx] * y[p * incy];
    }
    *gamma += alpha * sum;
}

//------------------------------------------------------
// Leftover-column kernels: compute one column of C for a packed MR-row strip,
// SIMD across the rows. These replace the scalar per-(row,col) fallback for the
// < NR=12 trailing columns. That fallback used to dominate at sizes where n is
// not a multiple of 12 (e.g. powers of two: n%12 = 4 or 8), dragging a kernel
// that otherwise matches OpenBLAS down by ~20%. 'a' is the packed A strip
// (MR floats per k step); 'bcol' is the raw B column (k elements, stride bstr);
// 'c' points at &C(col,row) so the rows are at c[r*ldc].
//------------------------------------------------------
static void AddDot8x1col_neon(CBLAS_INDEX k, float *a, float *bcol, CBLAS_INDEX bstr,
                              float *c, CBLAS_INDEX ldc, float alpha)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        float bj = bcol[p * bstr];
        acc0 = vfmaq_n_f32(acc0, vld1q_f32(a + p * 8),     bj);
        acc1 = vfmaq_n_f32(acc1, vld1q_f32(a + p * 8 + 4), bj);
    }
    float r[8];
    vst1q_f32(r,     acc0);
    vst1q_f32(r + 4, acc1);
    for (int i = 0; i < 8; i++)
        c[i * ldc] += alpha * r[i];
}

static void AddDot4x1col_neon(CBLAS_INDEX k, float *a, float *bcol, CBLAS_INDEX bstr,
                              float *c, CBLAS_INDEX ldc, float alpha)
{
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (CBLAS_INDEX p = 0; p < k; p++)
        acc = vfmaq_n_f32(acc, vld1q_f32(a + p * 4), bcol[p * bstr]);
    float r[4];
    vst1q_f32(r, acc);
    for (int i = 0; i < 4; i++)
        c[i * ldc] += alpha * r[i];
}

//------------------------------------------------------
// 8x12 micro-kernel using NEON with vfmaq_laneq_f32
// Computes C[8x12] += alpha * A[8xk] * B[kx12]
// Uses 24 float32x4_t registers for C (8 rows × 3 per row)
// Uses vfmaq_laneq_f32 for efficient A element broadcasting
// K-loop unrolled by 4 for better ILP
//------------------------------------------------------
static void AddDot8x12_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 8 rows × 3 float32x4_t = 12 columns
    float32x4_t c00, c01, c02;  // Row 0: columns 0-3, 4-7, 8-11
    float32x4_t c10, c11, c12;  // Row 1
    float32x4_t c20, c21, c22;  // Row 2
    float32x4_t c30, c31, c32;  // Row 3
    float32x4_t c40, c41, c42;  // Row 4
    float32x4_t c50, c51, c52;  // Row 5
    float32x4_t c60, c61, c62;  // Row 6
    float32x4_t c70, c71, c72;  // Row 7
    
    float32x4_t b0, b1, b2;     // B row: 12 floats = 3 float32x4_t
    float32x4_t a0, a1;         // A elements: 8 floats = 2 float32x4_t
    
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    // Initialize accumulators to zero
    c00 = vdupq_n_f32(0.0f); c01 = vdupq_n_f32(0.0f); c02 = vdupq_n_f32(0.0f);
    c10 = vdupq_n_f32(0.0f); c11 = vdupq_n_f32(0.0f); c12 = vdupq_n_f32(0.0f);
    c20 = vdupq_n_f32(0.0f); c21 = vdupq_n_f32(0.0f); c22 = vdupq_n_f32(0.0f);
    c30 = vdupq_n_f32(0.0f); c31 = vdupq_n_f32(0.0f); c32 = vdupq_n_f32(0.0f);
    c40 = vdupq_n_f32(0.0f); c41 = vdupq_n_f32(0.0f); c42 = vdupq_n_f32(0.0f);
    c50 = vdupq_n_f32(0.0f); c51 = vdupq_n_f32(0.0f); c52 = vdupq_n_f32(0.0f);
    c60 = vdupq_n_f32(0.0f); c61 = vdupq_n_f32(0.0f); c62 = vdupq_n_f32(0.0f);
    c70 = vdupq_n_f32(0.0f); c71 = vdupq_n_f32(0.0f); c72 = vdupq_n_f32(0.0f);
    
    CBLAS_INDEX p = 0;
    
    // Main loop: unrolled by 4 for better ILP
    // Process 4 k-iterations per loop to reduce overhead and improve pipelining
    for (; p + 4 <= k; p += 4)
    {
        // Iteration 0
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b2 = vld1q_f32(b + 8);
        a0 = vld1q_f32(a);
        a1 = vld1q_f32(a + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        
        // Iteration 1
        b0 = vld1q_f32(b + NR);
        b1 = vld1q_f32(b + NR + 4);
        b2 = vld1q_f32(b + NR + 8);
        a0 = vld1q_f32(a + MR);
        a1 = vld1q_f32(a + MR + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        
        // Iteration 2
        b0 = vld1q_f32(b + 2*NR);
        b1 = vld1q_f32(b + 2*NR + 4);
        b2 = vld1q_f32(b + 2*NR + 8);
        a0 = vld1q_f32(a + 2*MR);
        a1 = vld1q_f32(a + 2*MR + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        
        // Iteration 3
        b0 = vld1q_f32(b + 3*NR);
        b1 = vld1q_f32(b + 3*NR + 4);
        b2 = vld1q_f32(b + 3*NR + 8);
        a0 = vld1q_f32(a + 3*MR);
        a1 = vld1q_f32(a + 3*MR + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        
        b += 4 * NR;
        a += 4 * MR;
    }
    
    // Handle remaining k iterations (0-3)
    for (; p < k; p++)
    {
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b2 = vld1q_f32(b + 8);
        a0 = vld1q_f32(a);
        a1 = vld1q_f32(a + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);
        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);
        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);
        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);
        
        b += NR;
        a += MR;
    }
    
    // Load old C, apply alpha, accumulate, store
    float32x4_t c_old0, c_old1, c_old2;
    
    // Row 0
    c_old0 = vld1q_f32(&C(0, 0));
    c_old1 = vld1q_f32(&C(4, 0));
    c_old2 = vld1q_f32(&C(8, 0));
    c00 = vfmaq_f32(c_old0, alpha_vec, c00);
    c01 = vfmaq_f32(c_old1, alpha_vec, c01);
    c02 = vfmaq_f32(c_old2, alpha_vec, c02);
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
    vst1q_f32(&C(8, 0), c02);
    
    // Row 1
    c_old0 = vld1q_f32(&C(0, 1));
    c_old1 = vld1q_f32(&C(4, 1));
    c_old2 = vld1q_f32(&C(8, 1));
    c10 = vfmaq_f32(c_old0, alpha_vec, c10);
    c11 = vfmaq_f32(c_old1, alpha_vec, c11);
    c12 = vfmaq_f32(c_old2, alpha_vec, c12);
    vst1q_f32(&C(0, 1), c10);
    vst1q_f32(&C(4, 1), c11);
    vst1q_f32(&C(8, 1), c12);
    
    // Row 2
    c_old0 = vld1q_f32(&C(0, 2));
    c_old1 = vld1q_f32(&C(4, 2));
    c_old2 = vld1q_f32(&C(8, 2));
    c20 = vfmaq_f32(c_old0, alpha_vec, c20);
    c21 = vfmaq_f32(c_old1, alpha_vec, c21);
    c22 = vfmaq_f32(c_old2, alpha_vec, c22);
    vst1q_f32(&C(0, 2), c20);
    vst1q_f32(&C(4, 2), c21);
    vst1q_f32(&C(8, 2), c22);
    
    // Row 3
    c_old0 = vld1q_f32(&C(0, 3));
    c_old1 = vld1q_f32(&C(4, 3));
    c_old2 = vld1q_f32(&C(8, 3));
    c30 = vfmaq_f32(c_old0, alpha_vec, c30);
    c31 = vfmaq_f32(c_old1, alpha_vec, c31);
    c32 = vfmaq_f32(c_old2, alpha_vec, c32);
    vst1q_f32(&C(0, 3), c30);
    vst1q_f32(&C(4, 3), c31);
    vst1q_f32(&C(8, 3), c32);
    
    // Row 4
    c_old0 = vld1q_f32(&C(0, 4));
    c_old1 = vld1q_f32(&C(4, 4));
    c_old2 = vld1q_f32(&C(8, 4));
    c40 = vfmaq_f32(c_old0, alpha_vec, c40);
    c41 = vfmaq_f32(c_old1, alpha_vec, c41);
    c42 = vfmaq_f32(c_old2, alpha_vec, c42);
    vst1q_f32(&C(0, 4), c40);
    vst1q_f32(&C(4, 4), c41);
    vst1q_f32(&C(8, 4), c42);
    
    // Row 5
    c_old0 = vld1q_f32(&C(0, 5));
    c_old1 = vld1q_f32(&C(4, 5));
    c_old2 = vld1q_f32(&C(8, 5));
    c50 = vfmaq_f32(c_old0, alpha_vec, c50);
    c51 = vfmaq_f32(c_old1, alpha_vec, c51);
    c52 = vfmaq_f32(c_old2, alpha_vec, c52);
    vst1q_f32(&C(0, 5), c50);
    vst1q_f32(&C(4, 5), c51);
    vst1q_f32(&C(8, 5), c52);
    
    // Row 6
    c_old0 = vld1q_f32(&C(0, 6));
    c_old1 = vld1q_f32(&C(4, 6));
    c_old2 = vld1q_f32(&C(8, 6));
    c60 = vfmaq_f32(c_old0, alpha_vec, c60);
    c61 = vfmaq_f32(c_old1, alpha_vec, c61);
    c62 = vfmaq_f32(c_old2, alpha_vec, c62);
    vst1q_f32(&C(0, 6), c60);
    vst1q_f32(&C(4, 6), c61);
    vst1q_f32(&C(8, 6), c62);
    
    // Row 7
    c_old0 = vld1q_f32(&C(0, 7));
    c_old1 = vld1q_f32(&C(4, 7));
    c_old2 = vld1q_f32(&C(8, 7));
    c70 = vfmaq_f32(c_old0, alpha_vec, c70);
    c71 = vfmaq_f32(c_old1, alpha_vec, c71);
    c72 = vfmaq_f32(c_old2, alpha_vec, c72);
    vst1q_f32(&C(0, 7), c70);
    vst1q_f32(&C(4, 7), c71);
    vst1q_f32(&C(8, 7), c72);
}

//------------------------------------------------------
// 4x12 micro-kernel for remainder handling (4 <= rows < 8)
// K-loop unrolled by 2 for better ILP
//------------------------------------------------------
static void AddDot4x12_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    float32x4_t c00, c01, c02;
    float32x4_t c10, c11, c12;
    float32x4_t c20, c21, c22;
    float32x4_t c30, c31, c32;
    
    float32x4_t b0, b1, b2;
    float32x4_t a0;
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    c00 = vdupq_n_f32(0.0f); c01 = vdupq_n_f32(0.0f); c02 = vdupq_n_f32(0.0f);
    c10 = vdupq_n_f32(0.0f); c11 = vdupq_n_f32(0.0f); c12 = vdupq_n_f32(0.0f);
    c20 = vdupq_n_f32(0.0f); c21 = vdupq_n_f32(0.0f); c22 = vdupq_n_f32(0.0f);
    c30 = vdupq_n_f32(0.0f); c31 = vdupq_n_f32(0.0f); c32 = vdupq_n_f32(0.0f);
    
    CBLAS_INDEX p = 0;
    
    // Unrolled by 2
    for (; p + 2 <= k; p += 2)
    {
        // Iteration 0
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b2 = vld1q_f32(b + 8);
        a0 = vld1q_f32(a);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        
        // Iteration 1
        b0 = vld1q_f32(b + NR);
        b1 = vld1q_f32(b + NR + 4);
        b2 = vld1q_f32(b + NR + 8);
        a0 = vld1q_f32(a + 4);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        
        b += 2 * NR;
        a += 2 * 4;
    }
    
    // Handle remaining iteration
    for (; p < k; p++)
    {
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b2 = vld1q_f32(b + 8);
        a0 = vld1q_f32(a);
        
        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        
        b += NR;
        a += 4;
    }
    
    float32x4_t c_old0, c_old1, c_old2;
    
    c_old0 = vld1q_f32(&C(0, 0)); c_old1 = vld1q_f32(&C(4, 0)); c_old2 = vld1q_f32(&C(8, 0));
    vst1q_f32(&C(0, 0), vfmaq_f32(c_old0, alpha_vec, c00));
    vst1q_f32(&C(4, 0), vfmaq_f32(c_old1, alpha_vec, c01));
    vst1q_f32(&C(8, 0), vfmaq_f32(c_old2, alpha_vec, c02));
    
    c_old0 = vld1q_f32(&C(0, 1)); c_old1 = vld1q_f32(&C(4, 1)); c_old2 = vld1q_f32(&C(8, 1));
    vst1q_f32(&C(0, 1), vfmaq_f32(c_old0, alpha_vec, c10));
    vst1q_f32(&C(4, 1), vfmaq_f32(c_old1, alpha_vec, c11));
    vst1q_f32(&C(8, 1), vfmaq_f32(c_old2, alpha_vec, c12));
    
    c_old0 = vld1q_f32(&C(0, 2)); c_old1 = vld1q_f32(&C(4, 2)); c_old2 = vld1q_f32(&C(8, 2));
    vst1q_f32(&C(0, 2), vfmaq_f32(c_old0, alpha_vec, c20));
    vst1q_f32(&C(4, 2), vfmaq_f32(c_old1, alpha_vec, c21));
    vst1q_f32(&C(8, 2), vfmaq_f32(c_old2, alpha_vec, c22));
    
    c_old0 = vld1q_f32(&C(0, 3)); c_old1 = vld1q_f32(&C(4, 3)); c_old2 = vld1q_f32(&C(8, 3));
    vst1q_f32(&C(0, 3), vfmaq_f32(c_old0, alpha_vec, c30));
    vst1q_f32(&C(4, 3), vfmaq_f32(c_old1, alpha_vec, c31));
    vst1q_f32(&C(8, 3), vfmaq_f32(c_old2, alpha_vec, c32));
}

//------------------------------------------------------
// 4x8 micro-kernel using NEON (legacy, kept for potential future use)
// Computes C[4x8] += alpha * A[4xk] * B[kx8]
// Uses 8 float32x4_t registers for C (4 rows × 2 per row)
//------------------------------------------------------
CBLAS_UNUSED static void AddDot4x8_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 4 rows × 2 float32x4_t = 8 columns
    float32x4_t c00, c01;  // Row 0: columns 0-3, 4-7
    float32x4_t c10, c11;  // Row 1
    float32x4_t c20, c21;  // Row 2
    float32x4_t c30, c31;  // Row 3
    
    float32x4_t b0, b1;    // B row: 8 floats = 2 float32x4_t
    float32x4_t a_elem;    // A element broadcast
    
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    // Initialize accumulators to zero
    c00 = vdupq_n_f32(0.0f); c01 = vdupq_n_f32(0.0f);
    c10 = vdupq_n_f32(0.0f); c11 = vdupq_n_f32(0.0f);
    c20 = vdupq_n_f32(0.0f); c21 = vdupq_n_f32(0.0f);
    c30 = vdupq_n_f32(0.0f); c31 = vdupq_n_f32(0.0f);
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (8 floats from packed format)
        b0 = vld1q_f32(b);      // B[p, 0:3]
        b1 = vld1q_f32(b + 4);  // B[p, 4:7]
        b += NR_LEGACY;  // 8 columns
        
#ifdef __ARM_FEATURE_FMA
        // Row 0
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vfmaq_f32(c00, a_elem, b0);
        c01 = vfmaq_f32(c01, a_elem, b1);
        
        // Row 1
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vfmaq_f32(c10, a_elem, b0);
        c11 = vfmaq_f32(c11, a_elem, b1);
        
        // Row 2
        a_elem = vld1q_dup_f32(&a[2]);
        c20 = vfmaq_f32(c20, a_elem, b0);
        c21 = vfmaq_f32(c21, a_elem, b1);
        
        // Row 3
        a_elem = vld1q_dup_f32(&a[3]);
        c30 = vfmaq_f32(c30, a_elem, b0);
        c31 = vfmaq_f32(c31, a_elem, b1);
#else
        // Row 0
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vaddq_f32(c00, vmulq_f32(a_elem, b0));
        c01 = vaddq_f32(c01, vmulq_f32(a_elem, b1));
        
        // Row 1
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vaddq_f32(c10, vmulq_f32(a_elem, b0));
        c11 = vaddq_f32(c11, vmulq_f32(a_elem, b1));
        
        // Row 2
        a_elem = vld1q_dup_f32(&a[2]);
        c20 = vaddq_f32(c20, vmulq_f32(a_elem, b0));
        c21 = vaddq_f32(c21, vmulq_f32(a_elem, b1));
        
        // Row 3
        a_elem = vld1q_dup_f32(&a[3]);
        c30 = vaddq_f32(c30, vmulq_f32(a_elem, b0));
        c31 = vaddq_f32(c31, vmulq_f32(a_elem, b1));
#endif
        
        a += MR_LEGACY;  // 4 rows
    }
    
    // Load old C, apply alpha, accumulate, store
    float32x4_t c_old0, c_old1;
    
#ifdef __ARM_FEATURE_FMA
    // Row 0
    c_old0 = vld1q_f32(&C(0, 0));
    c_old1 = vld1q_f32(&C(4, 0));
    c00 = vfmaq_f32(c_old0, alpha_vec, c00);
    c01 = vfmaq_f32(c_old1, alpha_vec, c01);
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
    
    // Row 1
    c_old0 = vld1q_f32(&C(0, 1));
    c_old1 = vld1q_f32(&C(4, 1));
    c10 = vfmaq_f32(c_old0, alpha_vec, c10);
    c11 = vfmaq_f32(c_old1, alpha_vec, c11);
    vst1q_f32(&C(0, 1), c10);
    vst1q_f32(&C(4, 1), c11);
    
    // Row 2
    c_old0 = vld1q_f32(&C(0, 2));
    c_old1 = vld1q_f32(&C(4, 2));
    c20 = vfmaq_f32(c_old0, alpha_vec, c20);
    c21 = vfmaq_f32(c_old1, alpha_vec, c21);
    vst1q_f32(&C(0, 2), c20);
    vst1q_f32(&C(4, 2), c21);
    
    // Row 3
    c_old0 = vld1q_f32(&C(0, 3));
    c_old1 = vld1q_f32(&C(4, 3));
    c30 = vfmaq_f32(c_old0, alpha_vec, c30);
    c31 = vfmaq_f32(c_old1, alpha_vec, c31);
    vst1q_f32(&C(0, 3), c30);
    vst1q_f32(&C(4, 3), c31);
#else
    // Row 0
    c_old0 = vld1q_f32(&C(0, 0));
    c_old1 = vld1q_f32(&C(4, 0));
    c00 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c00));
    c01 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c01));
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
    
    // Row 1
    c_old0 = vld1q_f32(&C(0, 1));
    c_old1 = vld1q_f32(&C(4, 1));
    c10 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c10));
    c11 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c11));
    vst1q_f32(&C(0, 1), c10);
    vst1q_f32(&C(4, 1), c11);
    
    // Row 2
    c_old0 = vld1q_f32(&C(0, 2));
    c_old1 = vld1q_f32(&C(4, 2));
    c20 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c20));
    c21 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c21));
    vst1q_f32(&C(0, 2), c20);
    vst1q_f32(&C(4, 2), c21);
    
    // Row 3
    c_old0 = vld1q_f32(&C(0, 3));
    c_old1 = vld1q_f32(&C(4, 3));
    c30 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c30));
    c31 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c31));
    vst1q_f32(&C(0, 3), c30);
    vst1q_f32(&C(4, 3), c31);
#endif
}

//------------------------------------------------------
// PackMatrixA_8 - Copy a 8×k panel of A into contiguous memory
// For 8x12 micro-kernel
//------------------------------------------------------
static void PackMatrixA_8(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_ptrs[8];
    for (CBLAS_INDEX r = 0; r < MR; r++) {
        a_ptrs[r] = (r < m_rows) ? &A(0, r) : NULL;
    }
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        for (CBLAS_INDEX r = 0; r < MR; r++) {
            a_to[r] = a_ptrs[r] ? *a_ptrs[r]++ : 0.0f;
        }
        a_to += MR;
    }
}

//------------------------------------------------------
// PackMatrixA_8_trans - Pack 8×k panel of A from transposed storage
// When A is transposed, logical A[row+r, p] = a[p*lda + r]
//------------------------------------------------------
static void PackMatrixA_8_trans(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *a_col = a + i * lda;
        
        for (CBLAS_INDEX r = 0; r < MR; r++) {
            a_to[r] = (r < m_rows) ? a_col[r] : 0.0f;
        }
        
        a_to += MR;
    }
}

//------------------------------------------------------
// PackMatrixA_4 - Copy a 4×k panel of A into contiguous memory
// For 4x12 remainder kernel
//------------------------------------------------------
static void PackMatrixA_4(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_ptrs[4];
    for (CBLAS_INDEX r = 0; r < 4; r++) {
        a_ptrs[r] = (r < m_rows) ? &A(0, r) : NULL;
    }
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        for (CBLAS_INDEX r = 0; r < 4; r++) {
            a_to[r] = a_ptrs[r] ? *a_ptrs[r]++ : 0.0f;
        }
        a_to += 4;
    }
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
        
        for (CBLAS_INDEX r = 0; r < 4; r++) {
            a_to[r] = (r < m_rows) ? a_col[r] : 0.0f;
        }
        
        a_to += 4;
    }
}

//------------------------------------------------------
// PackMatrixA_2 - Pack 2 rows (kept for potential future use)
//------------------------------------------------------
CBLAS_UNUSED static void PackMatrixA_2_neon(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
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
// PackMatrixA_1 - Pack 1 row (kept for potential future use)
//------------------------------------------------------
CBLAS_UNUSED static void PackMatrixA_1_neon(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_0 = &A(0,0);
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[i] = *a_0++;
    }
}

//------------------------------------------------------
// PackMatrixB_12 - Copy a k×12 panel of B into contiguous memory
// Vectorized version using NEON for full-width copies
//------------------------------------------------------
static void PackMatrixB_12(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    if (n_cols >= NR) {
        // Fast path: full 12 columns, use NEON loads/stores
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            
            // Load and store 12 floats using NEON (3 float32x4_t registers)
            float32x4_t b0 = vld1q_f32(b_ij_pntr);
            float32x4_t b1 = vld1q_f32(b_ij_pntr + 4);
            float32x4_t b2 = vld1q_f32(b_ij_pntr + 8);
            vst1q_f32(b_to, b0);
            vst1q_f32(b_to + 4, b1);
            vst1q_f32(b_to + 8, b2);
            
            b_to += NR;
        }
    } else {
        // Slow path: partial columns, need zero-padding
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            CBLAS_INDEX col;
            for (col = 0; col < n_cols; col++) {
                b_to[col] = b_ij_pntr[col];
            }
            for (; col < NR; col++) {
                b_to[col] = 0.0f;
            }
            b_to += NR;
        }
    }
}

//------------------------------------------------------
// PackMatrixB_12_trans - Pack k×12 panel of B from transposed storage
// When B is transposed, logical B[p, col+c] = b[c*ldb + p]
//------------------------------------------------------
static void PackMatrixB_12_trans(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
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
// PackMatrixB_8 - Copy a k×8 panel of B (kept for potential future use)
//------------------------------------------------------
CBLAS_UNUSED static void PackMatrixB_8(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    if (n_cols >= NR_LEGACY) {
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            float32x4_t b0 = vld1q_f32(b_ij_pntr);
            float32x4_t b1 = vld1q_f32(b_ij_pntr + 4);
            vst1q_f32(b_to, b0);
            vst1q_f32(b_to + 4, b1);
            b_to += NR_LEGACY;
        }
    } else {
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            CBLAS_INDEX col;
            for (col = 0; col < n_cols; col++) {
                b_to[col] = b_ij_pntr[col];
            }
            for (; col < NR_LEGACY; col++) {
                b_to[col] = 0.0f;
            }
            b_to += NR_LEGACY;
        }
    }
}

//------------------------------------------------------
// InnerKernel - NEON implementation with 8x12 micro-kernel
// Pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_neon(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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
            PackMatrixB_12_trans(k, NR, b_ptr, ldb, &packedB[col * k]);
        else
            PackMatrixB_12(k, NR, b_ptr, ldb, &packedB[col * k]);
    }

    // Main loop: 8 rows at a time with 8x12 kernel
    for (row = 0; row + MR <= m; row += MR)
    {
        // Pack this 8×k panel of A once per row iteration
        if (transA)
            PackMatrixA_8_trans(k, MR, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_8(k, MR, a + row * a_row_stride, lda, packedA);

        // Process 12 columns at a time using pre-packed B
        for (col = 0; col + NR <= n; col += NR)
        {
            AddDot8x12_neon(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 12) - SIMD across the 8 packed rows
        for (; col < n; col++)
            AddDot8x1col_neon(k, packedA, b + col * b_col_stride, b_row_stride, &C(col, row), ldc, alpha);
    }

    // Handle leftover rows (< 8) using smaller kernels
    CBLAS_INDEX remaining_rows = m - row;
    
    // Use 4x12 kernel for 4-7 remaining rows
    if (remaining_rows >= 4)
    {
        if (transA)
            PackMatrixA_4_trans(k, 4, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_4(k, 4, a + row * a_row_stride, lda, packedA);
        
        for (col = 0; col + NR <= n; col += NR)
        {
            AddDot4x12_neon(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);
        }
        
        // Leftover columns - SIMD across the 4 packed rows
        for (; col < n; col++)
            AddDot4x1col_neon(k, packedA, b + col * b_col_stride, b_row_stride, &C(col, row), ldc, alpha);
        row += 4;
        remaining_rows -= 4;
    }
    
    // Use scalar for remaining 1-3 rows
    for (; row < m; row++)
    {
        for (col = 0; col < n; col++) {
            AddDot(k, a + row * a_row_stride, a_col_stride, b + col * b_col_stride, b_row_stride, &C(col, row), alpha);
        }
    }
    
    if (!use_pool_a) free(packedA);
    if (!use_pool_b) free(packedB);
}

//------------------------------------------------------
// SGEMM kernel - NEON version (8x12 micro-kernel)
//------------------------------------------------------
void sgemm_k_neon(cblas_args_t* args)
{
    InnerKernel_neon(args->ib, args->n, args->pb,
                     args->a, args->lda,
                     args->b, args->ldb,
                     args->c, args->ldc,
                     args->alpha_s, args->thread_id,
                     args->transa, args->transb);
}

//------------------------------------------------------
// Cooperative MT SGEMM: pack a k x n panel of B (full NR=12 column panels)
// into the shared args->packedB buffer. Called once per kc-panel; every
// m-band macro-kernel then reads this packed B from the shared L2 instead of
// re-streaming raw B from memory. Leftover (< NR) columns are left to the
// macro-kernel's scalar path, which reads raw B directly.
//------------------------------------------------------
void sgemm_pack_b_neon(cblas_args_t* args)
{
    CBLAS_INDEX k = args->pb;       // kc for this panel
    CBLAS_INDEX n = args->n;        // full panel width
    CBLAS_INDEX ldb = args->ldb;
    float *b = (float*)args->b;
    float *packedB = (float*)args->packedB;
    int transB = (args->transb == CblasTrans);
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    for (CBLAS_INDEX col = 0; col + NR <= n; col += NR)
    {
        float *b_ptr = b + col * b_col_stride;
        if (transB)
            PackMatrixB_12_trans(k, NR, b_ptr, ldb, &packedB[col * k]);
        else
            PackMatrixB_12(k, NR, b_ptr, ldb, &packedB[col * k]);
    }
}

//------------------------------------------------------
// Cooperative MT SGEMM macro-kernel: compute one m-band of C using the shared
// pre-packed B. Packs this band's A in MR=8 strips (per-thread buffer) and runs
// the 8x12 micro-kernel against the shared packedB; 4x12 + scalar handle the
// row remainder, scalar (reading raw B) handles the < NR column remainder.
//
// args: a=raw A band base, b=raw B panel base (for leftover cols),
//   packedB=shared packed B, c=C band base, ib=band rows, n=panel width,
//   pb=kc, lda/ldb/ldc, alpha_s, transa/transb, thread_id.
//------------------------------------------------------
void sgemm_macro_k_neon(cblas_args_t* args)
{
    CBLAS_INDEX m = args->ib;       // rows in this band
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX k = args->pb;       // kc
    CBLAS_INDEX lda = args->lda, ldb = args->ldb, ldc = args->ldc;
    float *a = (float*)args->a;
    float *b = (float*)args->b;             // raw B panel (leftover columns)
    float *c = (float*)args->c;
    float *packedB = (float*)args->packedB;
    float alpha = args->alpha_s;

    int transA = (args->transa == CblasTrans);
    int transB = (args->transb == CblasTrans);
    CBLAS_INDEX a_row_stride = transA ? 1 : lda;
    CBLAS_INDEX a_col_stride = transA ? lda : 1;
    CBLAS_INDEX b_row_stride = transB ? 1 : ldb;
    CBLAS_INDEX b_col_stride = transB ? ldb : 1;

    cblas_gemm_buffer_t* buf = cblas_get_gemm_buffer(args->thread_id);
    float *packedA;
    int use_pool_a = 0;
    if (buf) { packedA = buf->packedA_s; use_pool_a = 1; }
    else { packedA = (float*)malloc(MR * k * sizeof(float)); if (!packedA) return; }

    CBLAS_INDEX row, col;

    for (row = 0; row + MR <= m; row += MR)
    {
        if (transA)
            PackMatrixA_8_trans(k, MR, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_8(k, MR, a + row * a_row_stride, lda, packedA);

        for (col = 0; col + NR <= n; col += NR)
            AddDot8x12_neon(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);

        for (; col < n; col++)
            AddDot8x1col_neon(k, packedA, b + col * b_col_stride, b_row_stride, &C(col, row), ldc, alpha);
    }

    CBLAS_INDEX remaining_rows = m - row;
    if (remaining_rows >= 4)
    {
        if (transA)
            PackMatrixA_4_trans(k, 4, a + row * a_row_stride, lda, packedA);
        else
            PackMatrixA_4(k, 4, a + row * a_row_stride, lda, packedA);

        for (col = 0; col + NR <= n; col += NR)
            AddDot4x12_neon(k, packedA, &packedB[col * k], &C(col, row), ldc, alpha);

        for (; col < n; col++)
            AddDot4x1col_neon(k, packedA, b + col * b_col_stride, b_row_stride, &C(col, row), ldc, alpha);
        row += 4;
    }

    for (; row < m; row++)
        for (col = 0; col < n; col++)
            AddDot(k, a + row * a_row_stride, a_col_stride,
                   b + col * b_col_stride, b_row_stride, &C(col, row), alpha);

    if (!use_pool_a) free(packedA);
}

#endif // ARM64 NEON
