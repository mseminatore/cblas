//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - NEON implementation for ARM64
// Implements 8x12 micro-kernel for optimal ARM64 register utilization
//------------------------------------------------------

#include "cblas.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Micro-kernel dimensions for ARM64 NEON
// 8x12 = 24 float32x4_t registers for C (8 rows × 3 registers of 4 floats)
// ARM64 has 32 SIMD registers, leaving headroom for A, B loads
#define MR 8   // Rows per micro-kernel
#define NR 12  // Columns per micro-kernel (3 float32x4_t registers)

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
// 8x12 micro-kernel using NEON
// Computes C[8x12] += alpha * A[8xk] * B[kx12]
// Uses 24 float32x4_t registers for C (8 rows × 3 per row)
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
    float32x4_t a_elem;         // A element broadcast
    
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
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (12 floats from packed format)
        b0 = vld1q_f32(b);      // B[p, 0:3]
        b1 = vld1q_f32(b + 4);  // B[p, 4:7]
        b2 = vld1q_f32(b + 8);  // B[p, 8:11]
        b += NR;
        
        // Prefetch
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR), 0, 3);
        }
        
#ifdef __ARM_FEATURE_FMA
        // Row 0
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vfmaq_f32(c00, a_elem, b0);
        c01 = vfmaq_f32(c01, a_elem, b1);
        c02 = vfmaq_f32(c02, a_elem, b2);
        
        // Row 1
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vfmaq_f32(c10, a_elem, b0);
        c11 = vfmaq_f32(c11, a_elem, b1);
        c12 = vfmaq_f32(c12, a_elem, b2);
        
        // Row 2
        a_elem = vld1q_dup_f32(&a[2]);
        c20 = vfmaq_f32(c20, a_elem, b0);
        c21 = vfmaq_f32(c21, a_elem, b1);
        c22 = vfmaq_f32(c22, a_elem, b2);
        
        // Row 3
        a_elem = vld1q_dup_f32(&a[3]);
        c30 = vfmaq_f32(c30, a_elem, b0);
        c31 = vfmaq_f32(c31, a_elem, b1);
        c32 = vfmaq_f32(c32, a_elem, b2);
        
        // Row 4
        a_elem = vld1q_dup_f32(&a[4]);
        c40 = vfmaq_f32(c40, a_elem, b0);
        c41 = vfmaq_f32(c41, a_elem, b1);
        c42 = vfmaq_f32(c42, a_elem, b2);
        
        // Row 5
        a_elem = vld1q_dup_f32(&a[5]);
        c50 = vfmaq_f32(c50, a_elem, b0);
        c51 = vfmaq_f32(c51, a_elem, b1);
        c52 = vfmaq_f32(c52, a_elem, b2);
        
        // Row 6
        a_elem = vld1q_dup_f32(&a[6]);
        c60 = vfmaq_f32(c60, a_elem, b0);
        c61 = vfmaq_f32(c61, a_elem, b1);
        c62 = vfmaq_f32(c62, a_elem, b2);
        
        // Row 7
        a_elem = vld1q_dup_f32(&a[7]);
        c70 = vfmaq_f32(c70, a_elem, b0);
        c71 = vfmaq_f32(c71, a_elem, b1);
        c72 = vfmaq_f32(c72, a_elem, b2);
#else
        // Row 0
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vaddq_f32(c00, vmulq_f32(a_elem, b0));
        c01 = vaddq_f32(c01, vmulq_f32(a_elem, b1));
        c02 = vaddq_f32(c02, vmulq_f32(a_elem, b2));
        
        // Row 1
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vaddq_f32(c10, vmulq_f32(a_elem, b0));
        c11 = vaddq_f32(c11, vmulq_f32(a_elem, b1));
        c12 = vaddq_f32(c12, vmulq_f32(a_elem, b2));
        
        // Row 2
        a_elem = vld1q_dup_f32(&a[2]);
        c20 = vaddq_f32(c20, vmulq_f32(a_elem, b0));
        c21 = vaddq_f32(c21, vmulq_f32(a_elem, b1));
        c22 = vaddq_f32(c22, vmulq_f32(a_elem, b2));
        
        // Row 3
        a_elem = vld1q_dup_f32(&a[3]);
        c30 = vaddq_f32(c30, vmulq_f32(a_elem, b0));
        c31 = vaddq_f32(c31, vmulq_f32(a_elem, b1));
        c32 = vaddq_f32(c32, vmulq_f32(a_elem, b2));
        
        // Row 4
        a_elem = vld1q_dup_f32(&a[4]);
        c40 = vaddq_f32(c40, vmulq_f32(a_elem, b0));
        c41 = vaddq_f32(c41, vmulq_f32(a_elem, b1));
        c42 = vaddq_f32(c42, vmulq_f32(a_elem, b2));
        
        // Row 5
        a_elem = vld1q_dup_f32(&a[5]);
        c50 = vaddq_f32(c50, vmulq_f32(a_elem, b0));
        c51 = vaddq_f32(c51, vmulq_f32(a_elem, b1));
        c52 = vaddq_f32(c52, vmulq_f32(a_elem, b2));
        
        // Row 6
        a_elem = vld1q_dup_f32(&a[6]);
        c60 = vaddq_f32(c60, vmulq_f32(a_elem, b0));
        c61 = vaddq_f32(c61, vmulq_f32(a_elem, b1));
        c62 = vaddq_f32(c62, vmulq_f32(a_elem, b2));
        
        // Row 7
        a_elem = vld1q_dup_f32(&a[7]);
        c70 = vaddq_f32(c70, vmulq_f32(a_elem, b0));
        c71 = vaddq_f32(c71, vmulq_f32(a_elem, b1));
        c72 = vaddq_f32(c72, vmulq_f32(a_elem, b2));
#endif
        
        a += MR;
    }
    
    // Load old C, apply alpha, accumulate, store
    float32x4_t c_old0, c_old1, c_old2;
    
#ifdef __ARM_FEATURE_FMA
    #define STORE_ROW(row) \
        c_old0 = vld1q_f32(&C(0, row)); \
        c_old1 = vld1q_f32(&C(4, row)); \
        c_old2 = vld1q_f32(&C(8, row)); \
        c##row##0 = vfmaq_f32(c_old0, alpha_vec, c##row##0); \
        c##row##1 = vfmaq_f32(c_old1, alpha_vec, c##row##1); \
        c##row##2 = vfmaq_f32(c_old2, alpha_vec, c##row##2); \
        vst1q_f32(&C(0, row), c##row##0); \
        vst1q_f32(&C(4, row), c##row##1); \
        vst1q_f32(&C(8, row), c##row##2);
#else
    #define STORE_ROW(row) \
        c_old0 = vld1q_f32(&C(0, row)); \
        c_old1 = vld1q_f32(&C(4, row)); \
        c_old2 = vld1q_f32(&C(8, row)); \
        c##row##0 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c##row##0)); \
        c##row##1 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c##row##1)); \
        c##row##2 = vaddq_f32(c_old2, vmulq_f32(alpha_vec, c##row##2)); \
        vst1q_f32(&C(0, row), c##row##0); \
        vst1q_f32(&C(4, row), c##row##1); \
        vst1q_f32(&C(8, row), c##row##2);
#endif

    STORE_ROW(0)
    STORE_ROW(1)
    STORE_ROW(2)
    STORE_ROW(3)
    STORE_ROW(4)
    STORE_ROW(5)
    STORE_ROW(6)
    STORE_ROW(7)
    
    #undef STORE_ROW
}

//------------------------------------------------------
// PackMatrixB_12 - Copy a k×12 panel of B into contiguous memory
//------------------------------------------------------
static void PackMatrixB_12(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(0, j);
        
        if (j + 4 < k) {
            CBLAS_PREFETCH(&B(0, j + 4), 0, 3);
        }
        
        CBLAS_INDEX col;
        for (col = 0; col < n_cols && col < NR; col++) {
            b_to[col] = b_ij_pntr[col];
        }
        for (; col < NR; col++) {
            b_to[col] = 0.0f;
        }
        
        b_to += NR;
    }
}

//------------------------------------------------------
// PackMatrixA_8 - Copy a 8×k panel of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA_8(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_ptrs[8];
    
    for (CBLAS_INDEX r = 0; r < MR; r++) {
        a_ptrs[r] = (r < m_rows) ? &A(0, r) : NULL;
    }
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        if (i + 8 < k && a_ptrs[0]) {
            CBLAS_PREFETCH(a_ptrs[0] + 8, 0, 3);
        }
        
        for (CBLAS_INDEX r = 0; r < MR; r++) {
            a_to[r] = a_ptrs[r] ? *a_ptrs[r]++ : 0.0f;
        }
        
        a_to += MR;
    }
}

//------------------------------------------------------
// InnerKernel - NEON implementation with 8x12 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_neon(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                             float* a, CBLAS_INDEX lda, 
                             float* b, CBLAS_INDEX ldb, 
                             float* c, CBLAS_INDEX ldc,
                             float alpha)
{
    float* packedA = (float*)malloc(MR * k * sizeof(float));
    float* packedB = (float*)malloc(k * NR * sizeof(float));
    
    if (!packedA || !packedB) {
        free(packedA);
        free(packedB);
        return;
    }

    CBLAS_INDEX row, col;

    // Main loop: 8 rows at a time
    for (row = 0; row + MR <= m; row += MR)
    {
        // Pack this 8×k panel of A once per row iteration
        PackMatrixA_8(k, MR, &A(0, row), lda, packedA);

        // Process 12 columns at a time
        for (col = 0; col + NR <= n; col += NR)
        {
            // Pack this k×12 panel of B
            PackMatrixB_12(k, NR, &B(col, 0), ldb, packedB);
            
            // Call 8x12 micro-kernel
            AddDot8x12_neon(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 12) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR; r++) {
                AddDot(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows (< 8) with scalar
    for (; row < m; row++) {
        for (col = 0; col < n; col++) {
            AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
        }
    }
    
    free(packedA);
    free(packedB);
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
                     args->alpha_s);
}

#endif // ARM64 NEON
