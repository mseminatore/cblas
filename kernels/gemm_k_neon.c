//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - NEON implementation for ARM64
// Implements 4x8 micro-kernel (simpler, matches AVX structure)
//------------------------------------------------------

#include "cblas.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for ARM64 NEON
// 4x8 = 8 float32x4_t registers for C (4 rows × 2 registers of 4 floats)
#define MR 4   // Rows per micro-kernel
#define NR 8   // Columns per micro-kernel (2 float32x4_t registers)

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
// 4x8 micro-kernel using NEON
// Computes C[4x8] += alpha * A[4xk] * B[kx8]
// Uses 8 float32x4_t registers for C (4 rows × 2 per row)
//------------------------------------------------------
static void AddDot4x8_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
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
        b += NR;
        
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
        
        a += MR;
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
// 2x8 micro-kernel using NEON
// For remainder handling when 2 <= remaining_rows < 4
//------------------------------------------------------
static void AddDot2x8_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    float32x4_t c00, c01;  // Row 0
    float32x4_t c10, c11;  // Row 1
    
    float32x4_t b0, b1;
    float32x4_t a_elem;
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    c00 = vdupq_n_f32(0.0f); c01 = vdupq_n_f32(0.0f);
    c10 = vdupq_n_f32(0.0f); c11 = vdupq_n_f32(0.0f);
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b += NR;
        
#ifdef __ARM_FEATURE_FMA
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vfmaq_f32(c00, a_elem, b0);
        c01 = vfmaq_f32(c01, a_elem, b1);
        
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vfmaq_f32(c10, a_elem, b0);
        c11 = vfmaq_f32(c11, a_elem, b1);
#else
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vaddq_f32(c00, vmulq_f32(a_elem, b0));
        c01 = vaddq_f32(c01, vmulq_f32(a_elem, b1));
        
        a_elem = vld1q_dup_f32(&a[1]);
        c10 = vaddq_f32(c10, vmulq_f32(a_elem, b0));
        c11 = vaddq_f32(c11, vmulq_f32(a_elem, b1));
#endif
        
        a += 2;  // 2 rows packed
    }
    
    // Store results
    float32x4_t c_old0, c_old1;
    
#ifdef __ARM_FEATURE_FMA
    c_old0 = vld1q_f32(&C(0, 0));
    c_old1 = vld1q_f32(&C(4, 0));
    c00 = vfmaq_f32(c_old0, alpha_vec, c00);
    c01 = vfmaq_f32(c_old1, alpha_vec, c01);
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
    
    c_old0 = vld1q_f32(&C(0, 1));
    c_old1 = vld1q_f32(&C(4, 1));
    c10 = vfmaq_f32(c_old0, alpha_vec, c10);
    c11 = vfmaq_f32(c_old1, alpha_vec, c11);
    vst1q_f32(&C(0, 1), c10);
    vst1q_f32(&C(4, 1), c11);
#else
    c_old0 = vld1q_f32(&C(0, 0));
    c_old1 = vld1q_f32(&C(4, 0));
    c00 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c00));
    c01 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c01));
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
    
    c_old0 = vld1q_f32(&C(0, 1));
    c_old1 = vld1q_f32(&C(4, 1));
    c10 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c10));
    c11 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c11));
    vst1q_f32(&C(0, 1), c10);
    vst1q_f32(&C(4, 1), c11);
#endif
}

//------------------------------------------------------
// 1x8 micro-kernel using NEON
// For remainder handling when remaining_rows == 1
//------------------------------------------------------
static void AddDot1x8_neon(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    (void)ldc;  // Not used for single row
    
    float32x4_t c00, c01;  // Row 0
    float32x4_t b0, b1;
    float32x4_t a_elem;
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    
    c00 = vdupq_n_f32(0.0f);
    c01 = vdupq_n_f32(0.0f);
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = vld1q_f32(b);
        b1 = vld1q_f32(b + 4);
        b += NR;
        
#ifdef __ARM_FEATURE_FMA
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vfmaq_f32(c00, a_elem, b0);
        c01 = vfmaq_f32(c01, a_elem, b1);
#else
        a_elem = vld1q_dup_f32(&a[0]);
        c00 = vaddq_f32(c00, vmulq_f32(a_elem, b0));
        c01 = vaddq_f32(c01, vmulq_f32(a_elem, b1));
#endif
        
        a += 1;
    }
    
    // Store results
#ifdef __ARM_FEATURE_FMA
    float32x4_t c_old0 = vld1q_f32(&C(0, 0));
    float32x4_t c_old1 = vld1q_f32(&C(4, 0));
    c00 = vfmaq_f32(c_old0, alpha_vec, c00);
    c01 = vfmaq_f32(c_old1, alpha_vec, c01);
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
#else
    float32x4_t c_old0 = vld1q_f32(&C(0, 0));
    float32x4_t c_old1 = vld1q_f32(&C(4, 0));
    c00 = vaddq_f32(c_old0, vmulq_f32(alpha_vec, c00));
    c01 = vaddq_f32(c_old1, vmulq_f32(alpha_vec, c01));
    vst1q_f32(&C(0, 0), c00);
    vst1q_f32(&C(4, 0), c01);
#endif
}

//------------------------------------------------------
// PackMatrixA_2 - Pack 2 rows for use with 2x8 kernel
//------------------------------------------------------
static void PackMatrixA_2_neon(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
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
// PackMatrixA_1 - Pack 1 row for use with 1x8 kernel
//------------------------------------------------------
static void PackMatrixA_1_neon(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_0 = &A(0,0);
    
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        a_to[i] = *a_0++;
    }
}

//------------------------------------------------------
// PackMatrixB_8 - Copy a k×8 panel of B into contiguous memory
// Vectorized version using NEON for full-width copies
//------------------------------------------------------
static void PackMatrixB_8(CBLAS_INDEX k, CBLAS_INDEX n_cols, float *b, CBLAS_INDEX ldb, float *b_to)
{
    if (n_cols >= NR) {
        // Fast path: full 8 columns, use NEON loads/stores
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            float *b_ij_pntr = &B(0, j);
            
            // Load and store 8 floats using NEON (2 float32x4_t registers)
            float32x4_t b0 = vld1q_f32(b_ij_pntr);
            float32x4_t b1 = vld1q_f32(b_ij_pntr + 4);
            vst1q_f32(b_to, b0);
            vst1q_f32(b_to + 4, b1);
            
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
// PackMatrixA_4 - Copy a 4×k panel of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA_4(CBLAS_INDEX k, CBLAS_INDEX m_rows, float *a, CBLAS_INDEX lda, float *a_to)
{
    float *a_ptrs[4];
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
// InnerKernel - NEON implementation with 4x8 micro-kernel
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_neon(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
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

    // Main loop: 4 rows at a time
    for (row = 0; row + MR <= m; row += MR)
    {
        // Pack this 4×k panel of A once per row iteration
        PackMatrixA_4(k, MR, &A(0, row), lda, packedA);

        // Process 8 columns at a time
        for (col = 0; col + NR <= n; col += NR)
        {
            // Pack this k×8 panel of B
            PackMatrixB_8(k, NR, &B(col, 0), ldb, packedB);
            
            // Call 4x8 micro-kernel
            AddDot4x8_neon(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 8) - use scalar fallback
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR; r++) {
                AddDot(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows using optimized remainder kernels
    CBLAS_INDEX remaining_rows = m - row;
    if (remaining_rows > 0)
    {
        // Use 2x8 kernel for 2-3 remaining rows
        if (remaining_rows >= 2)
        {
            PackMatrixA_2_neon(k, &A(0, row), lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                PackMatrixB_8(k, NR, &B(col, 0), ldb, packedB);
                AddDot2x8_neon(k, packedA, packedB, &C(col, row), ldc, alpha);
            }
            
            // Leftover columns
            for (; col < n; col++) {
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row), alpha);
                AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1), alpha);
            }
            row += 2;
            remaining_rows -= 2;
        }
        
        // Use 1x8 kernel for last remaining row
        if (remaining_rows == 1)
        {
            PackMatrixA_1_neon(k, &A(0, row), lda, packedA);
            
            for (col = 0; col + NR <= n; col += NR)
            {
                PackMatrixB_8(k, NR, &B(col, 0), ldb, packedB);
                AddDot1x8_neon(k, packedA, packedB, &C(col, row), ldc, alpha);
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
// SGEMM kernel - NEON version (4x8 micro-kernel)
//------------------------------------------------------
void sgemm_k_neon(cblas_args_t* args)
{
    InnerKernel_neon(args->ib, args->n, args->pb, 
                     args->a, args->lda, 
                     args->b, args->ldb, 
                     args->c, args->ldc,
                     args->alpha_s, args->thread_id);
}

#endif // ARM64 NEON
