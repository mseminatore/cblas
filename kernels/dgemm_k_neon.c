//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// DGEMM kernel - NEON implementation for ARM64
// Implements 4x4 micro-kernel using float64x2_t (2 doubles per register)
//------------------------------------------------------

#include "cblas.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 8

// Micro-kernel dimensions for ARM64 NEON double precision
// 4x8 = 16 float64x2_t registers for C (4 rows × 4 registers of 2 doubles)
#define MR_D 4   // Rows per micro-kernel
#define NR_D 8   // Columns per micro-kernel (4 float64x2_t registers)

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
// 4x4 micro-kernel using NEON for double precision
// Computes C[4x4] += alpha * A[4xk] * B[kx4]
// Uses 8 float64x2_t registers for C (4 rows × 2 per row)
//------------------------------------------------------
static void AddDot4x4_dgemm_neon(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    // C accumulator registers: 4 rows × 2 float64x2_t = 4 columns
    float64x2_t c00, c01;  // Row 0: columns 0-1, 2-3
    float64x2_t c10, c11;  // Row 1
    float64x2_t c20, c21;  // Row 2
    float64x2_t c30, c31;  // Row 3
    
    float64x2_t b0, b1;    // B row: 4 doubles = 2 float64x2_t
    float64x2_t a_elem;    // A element broadcast
    
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    
    // Initialize accumulators to zero
    c00 = vdupq_n_f64(0.0); c01 = vdupq_n_f64(0.0);
    c10 = vdupq_n_f64(0.0); c11 = vdupq_n_f64(0.0);
    c20 = vdupq_n_f64(0.0); c21 = vdupq_n_f64(0.0);
    c30 = vdupq_n_f64(0.0); c31 = vdupq_n_f64(0.0);
    
    // Main loop over k dimension
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        // Load B row (4 doubles from packed format)
        b0 = vld1q_f64(b);      // B[p, 0:1]
        b1 = vld1q_f64(b + 2);  // B[p, 2:3]
        b += NR_D;
        
        // Prefetch
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * MR_D), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * NR_D), 0, 3);
        }
        
#ifdef __ARM_FEATURE_FMA
        // Row 0
        a_elem = vld1q_dup_f64(&a[0]);
        c00 = vfmaq_f64(c00, a_elem, b0);
        c01 = vfmaq_f64(c01, a_elem, b1);
        
        // Row 1
        a_elem = vld1q_dup_f64(&a[1]);
        c10 = vfmaq_f64(c10, a_elem, b0);
        c11 = vfmaq_f64(c11, a_elem, b1);
        
        // Row 2
        a_elem = vld1q_dup_f64(&a[2]);
        c20 = vfmaq_f64(c20, a_elem, b0);
        c21 = vfmaq_f64(c21, a_elem, b1);
        
        // Row 3
        a_elem = vld1q_dup_f64(&a[3]);
        c30 = vfmaq_f64(c30, a_elem, b0);
        c31 = vfmaq_f64(c31, a_elem, b1);
#else
        // Row 0
        a_elem = vld1q_dup_f64(&a[0]);
        c00 = vaddq_f64(c00, vmulq_f64(a_elem, b0));
        c01 = vaddq_f64(c01, vmulq_f64(a_elem, b1));
        
        // Row 1
        a_elem = vld1q_dup_f64(&a[1]);
        c10 = vaddq_f64(c10, vmulq_f64(a_elem, b0));
        c11 = vaddq_f64(c11, vmulq_f64(a_elem, b1));
        
        // Row 2
        a_elem = vld1q_dup_f64(&a[2]);
        c20 = vaddq_f64(c20, vmulq_f64(a_elem, b0));
        c21 = vaddq_f64(c21, vmulq_f64(a_elem, b1));
        
        // Row 3
        a_elem = vld1q_dup_f64(&a[3]);
        c30 = vaddq_f64(c30, vmulq_f64(a_elem, b0));
        c31 = vaddq_f64(c31, vmulq_f64(a_elem, b1));
#endif
        
        a += MR_D;
    }
    
    // Load old C, apply alpha, accumulate, store
    float64x2_t c_old0, c_old1;
    
#ifdef __ARM_FEATURE_FMA
    // Row 0
    c_old0 = vld1q_f64(&C(0, 0));
    c_old1 = vld1q_f64(&C(2, 0));
    c00 = vfmaq_f64(c_old0, alpha_vec, c00);
    c01 = vfmaq_f64(c_old1, alpha_vec, c01);
    vst1q_f64(&C(0, 0), c00);
    vst1q_f64(&C(2, 0), c01);
    
    // Row 1
    c_old0 = vld1q_f64(&C(0, 1));
    c_old1 = vld1q_f64(&C(2, 1));
    c10 = vfmaq_f64(c_old0, alpha_vec, c10);
    c11 = vfmaq_f64(c_old1, alpha_vec, c11);
    vst1q_f64(&C(0, 1), c10);
    vst1q_f64(&C(2, 1), c11);
    
    // Row 2
    c_old0 = vld1q_f64(&C(0, 2));
    c_old1 = vld1q_f64(&C(2, 2));
    c20 = vfmaq_f64(c_old0, alpha_vec, c20);
    c21 = vfmaq_f64(c_old1, alpha_vec, c21);
    vst1q_f64(&C(0, 2), c20);
    vst1q_f64(&C(2, 2), c21);
    
    // Row 3
    c_old0 = vld1q_f64(&C(0, 3));
    c_old1 = vld1q_f64(&C(2, 3));
    c30 = vfmaq_f64(c_old0, alpha_vec, c30);
    c31 = vfmaq_f64(c_old1, alpha_vec, c31);
    vst1q_f64(&C(0, 3), c30);
    vst1q_f64(&C(2, 3), c31);
#else
    // Row 0
    c_old0 = vld1q_f64(&C(0, 0));
    c_old1 = vld1q_f64(&C(2, 0));
    c00 = vaddq_f64(c_old0, vmulq_f64(alpha_vec, c00));
    c01 = vaddq_f64(c_old1, vmulq_f64(alpha_vec, c01));
    vst1q_f64(&C(0, 0), c00);
    vst1q_f64(&C(2, 0), c01);
    
    // Row 1
    c_old0 = vld1q_f64(&C(0, 1));
    c_old1 = vld1q_f64(&C(2, 1));
    c10 = vaddq_f64(c_old0, vmulq_f64(alpha_vec, c10));
    c11 = vaddq_f64(c_old1, vmulq_f64(alpha_vec, c11));
    vst1q_f64(&C(0, 1), c10);
    vst1q_f64(&C(2, 1), c11);
    
    // Row 2
    c_old0 = vld1q_f64(&C(0, 2));
    c_old1 = vld1q_f64(&C(2, 2));
    c20 = vaddq_f64(c_old0, vmulq_f64(alpha_vec, c20));
    c21 = vaddq_f64(c_old1, vmulq_f64(alpha_vec, c21));
    vst1q_f64(&C(0, 2), c20);
    vst1q_f64(&C(2, 2), c21);
    
    // Row 3
    c_old0 = vld1q_f64(&C(0, 3));
    c_old1 = vld1q_f64(&C(2, 3));
    c30 = vaddq_f64(c_old0, vmulq_f64(alpha_vec, c30));
    c31 = vaddq_f64(c_old1, vmulq_f64(alpha_vec, c31));
    vst1q_f64(&C(0, 3), c30);
    vst1q_f64(&C(2, 3), c31);
#endif
}

//------------------------------------------------------
// 4x8 micro-kernel using NEON for double precision
// Computes C[4x8] += alpha * A[4xk] * B[kx8]
// Uses 16 float64x2_t registers for C (4 rows × 4 per row)
// K-loop unrolled by 2 for better ILP
//------------------------------------------------------
static void AddDot4x8_dgemm_neon(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    // C accumulator registers: 4 rows × 4 float64x2_t = 8 columns
    float64x2_t c00, c01, c02, c03;  // Row 0: columns 0-1, 2-3, 4-5, 6-7
    float64x2_t c10, c11, c12, c13;  // Row 1
    float64x2_t c20, c21, c22, c23;  // Row 2
    float64x2_t c30, c31, c32, c33;  // Row 3
    
    float64x2_t b0, b1, b2, b3;      // B row: 8 doubles = 4 float64x2_t
    float64x2_t a_vec;               // A column: 2 doubles for lane broadcast
    
    float64x2_t alpha_vec = vdupq_n_f64(alpha);
    
    // Initialize accumulators to zero
    c00 = vdupq_n_f64(0.0); c01 = vdupq_n_f64(0.0); c02 = vdupq_n_f64(0.0); c03 = vdupq_n_f64(0.0);
    c10 = vdupq_n_f64(0.0); c11 = vdupq_n_f64(0.0); c12 = vdupq_n_f64(0.0); c13 = vdupq_n_f64(0.0);
    c20 = vdupq_n_f64(0.0); c21 = vdupq_n_f64(0.0); c22 = vdupq_n_f64(0.0); c23 = vdupq_n_f64(0.0);
    c30 = vdupq_n_f64(0.0); c31 = vdupq_n_f64(0.0); c32 = vdupq_n_f64(0.0); c33 = vdupq_n_f64(0.0);
    
    CBLAS_INDEX p = 0;
    
    // Main loop: unrolled by 2 for better ILP
    for (; p + 2 <= k; p += 2)
    {
        // Iteration 0
        b0 = vld1q_f64(b);
        b1 = vld1q_f64(b + 2);
        b2 = vld1q_f64(b + 4);
        b3 = vld1q_f64(b + 6);
        
        // Load A[0:1] for rows 0-1, use lane broadcast
        a_vec = vld1q_f64(a);
        c00 = vfmaq_laneq_f64(c00, b0, a_vec, 0);
        c01 = vfmaq_laneq_f64(c01, b1, a_vec, 0);
        c02 = vfmaq_laneq_f64(c02, b2, a_vec, 0);
        c03 = vfmaq_laneq_f64(c03, b3, a_vec, 0);
        c10 = vfmaq_laneq_f64(c10, b0, a_vec, 1);
        c11 = vfmaq_laneq_f64(c11, b1, a_vec, 1);
        c12 = vfmaq_laneq_f64(c12, b2, a_vec, 1);
        c13 = vfmaq_laneq_f64(c13, b3, a_vec, 1);
        
        // Load A[2:3] for rows 2-3
        a_vec = vld1q_f64(a + 2);
        c20 = vfmaq_laneq_f64(c20, b0, a_vec, 0);
        c21 = vfmaq_laneq_f64(c21, b1, a_vec, 0);
        c22 = vfmaq_laneq_f64(c22, b2, a_vec, 0);
        c23 = vfmaq_laneq_f64(c23, b3, a_vec, 0);
        c30 = vfmaq_laneq_f64(c30, b0, a_vec, 1);
        c31 = vfmaq_laneq_f64(c31, b1, a_vec, 1);
        c32 = vfmaq_laneq_f64(c32, b2, a_vec, 1);
        c33 = vfmaq_laneq_f64(c33, b3, a_vec, 1);
        
        // Iteration 1
        b0 = vld1q_f64(b + NR_D);
        b1 = vld1q_f64(b + NR_D + 2);
        b2 = vld1q_f64(b + NR_D + 4);
        b3 = vld1q_f64(b + NR_D + 6);
        
        a_vec = vld1q_f64(a + MR_D);
        c00 = vfmaq_laneq_f64(c00, b0, a_vec, 0);
        c01 = vfmaq_laneq_f64(c01, b1, a_vec, 0);
        c02 = vfmaq_laneq_f64(c02, b2, a_vec, 0);
        c03 = vfmaq_laneq_f64(c03, b3, a_vec, 0);
        c10 = vfmaq_laneq_f64(c10, b0, a_vec, 1);
        c11 = vfmaq_laneq_f64(c11, b1, a_vec, 1);
        c12 = vfmaq_laneq_f64(c12, b2, a_vec, 1);
        c13 = vfmaq_laneq_f64(c13, b3, a_vec, 1);
        
        a_vec = vld1q_f64(a + MR_D + 2);
        c20 = vfmaq_laneq_f64(c20, b0, a_vec, 0);
        c21 = vfmaq_laneq_f64(c21, b1, a_vec, 0);
        c22 = vfmaq_laneq_f64(c22, b2, a_vec, 0);
        c23 = vfmaq_laneq_f64(c23, b3, a_vec, 0);
        c30 = vfmaq_laneq_f64(c30, b0, a_vec, 1);
        c31 = vfmaq_laneq_f64(c31, b1, a_vec, 1);
        c32 = vfmaq_laneq_f64(c32, b2, a_vec, 1);
        c33 = vfmaq_laneq_f64(c33, b3, a_vec, 1);
        
        a += 2 * MR_D;
        b += 2 * NR_D;
    }
    
    // Handle remaining k iteration (if k is odd)
    for (; p < k; p++)
    {
        b0 = vld1q_f64(b);
        b1 = vld1q_f64(b + 2);
        b2 = vld1q_f64(b + 4);
        b3 = vld1q_f64(b + 6);
        
        a_vec = vld1q_f64(a);
        c00 = vfmaq_laneq_f64(c00, b0, a_vec, 0);
        c01 = vfmaq_laneq_f64(c01, b1, a_vec, 0);
        c02 = vfmaq_laneq_f64(c02, b2, a_vec, 0);
        c03 = vfmaq_laneq_f64(c03, b3, a_vec, 0);
        c10 = vfmaq_laneq_f64(c10, b0, a_vec, 1);
        c11 = vfmaq_laneq_f64(c11, b1, a_vec, 1);
        c12 = vfmaq_laneq_f64(c12, b2, a_vec, 1);
        c13 = vfmaq_laneq_f64(c13, b3, a_vec, 1);
        
        a_vec = vld1q_f64(a + 2);
        c20 = vfmaq_laneq_f64(c20, b0, a_vec, 0);
        c21 = vfmaq_laneq_f64(c21, b1, a_vec, 0);
        c22 = vfmaq_laneq_f64(c22, b2, a_vec, 0);
        c23 = vfmaq_laneq_f64(c23, b3, a_vec, 0);
        c30 = vfmaq_laneq_f64(c30, b0, a_vec, 1);
        c31 = vfmaq_laneq_f64(c31, b1, a_vec, 1);
        c32 = vfmaq_laneq_f64(c32, b2, a_vec, 1);
        c33 = vfmaq_laneq_f64(c33, b3, a_vec, 1);
        
        a += MR_D;
        b += NR_D;
    }
    
    // Load old C, apply alpha, accumulate, store
    float64x2_t c_old0, c_old1, c_old2, c_old3;
    
    // Row 0
    c_old0 = vld1q_f64(&C(0, 0));
    c_old1 = vld1q_f64(&C(2, 0));
    c_old2 = vld1q_f64(&C(4, 0));
    c_old3 = vld1q_f64(&C(6, 0));
    c00 = vfmaq_f64(c_old0, alpha_vec, c00);
    c01 = vfmaq_f64(c_old1, alpha_vec, c01);
    c02 = vfmaq_f64(c_old2, alpha_vec, c02);
    c03 = vfmaq_f64(c_old3, alpha_vec, c03);
    vst1q_f64(&C(0, 0), c00);
    vst1q_f64(&C(2, 0), c01);
    vst1q_f64(&C(4, 0), c02);
    vst1q_f64(&C(6, 0), c03);
    
    // Row 1
    c_old0 = vld1q_f64(&C(0, 1));
    c_old1 = vld1q_f64(&C(2, 1));
    c_old2 = vld1q_f64(&C(4, 1));
    c_old3 = vld1q_f64(&C(6, 1));
    c10 = vfmaq_f64(c_old0, alpha_vec, c10);
    c11 = vfmaq_f64(c_old1, alpha_vec, c11);
    c12 = vfmaq_f64(c_old2, alpha_vec, c12);
    c13 = vfmaq_f64(c_old3, alpha_vec, c13);
    vst1q_f64(&C(0, 1), c10);
    vst1q_f64(&C(2, 1), c11);
    vst1q_f64(&C(4, 1), c12);
    vst1q_f64(&C(6, 1), c13);
    
    // Row 2
    c_old0 = vld1q_f64(&C(0, 2));
    c_old1 = vld1q_f64(&C(2, 2));
    c_old2 = vld1q_f64(&C(4, 2));
    c_old3 = vld1q_f64(&C(6, 2));
    c20 = vfmaq_f64(c_old0, alpha_vec, c20);
    c21 = vfmaq_f64(c_old1, alpha_vec, c21);
    c22 = vfmaq_f64(c_old2, alpha_vec, c22);
    c23 = vfmaq_f64(c_old3, alpha_vec, c23);
    vst1q_f64(&C(0, 2), c20);
    vst1q_f64(&C(2, 2), c21);
    vst1q_f64(&C(4, 2), c22);
    vst1q_f64(&C(6, 2), c23);
    
    // Row 3
    c_old0 = vld1q_f64(&C(0, 3));
    c_old1 = vld1q_f64(&C(2, 3));
    c_old2 = vld1q_f64(&C(4, 3));
    c_old3 = vld1q_f64(&C(6, 3));
    c30 = vfmaq_f64(c_old0, alpha_vec, c30);
    c31 = vfmaq_f64(c_old1, alpha_vec, c31);
    c32 = vfmaq_f64(c_old2, alpha_vec, c32);
    c33 = vfmaq_f64(c_old3, alpha_vec, c33);
    vst1q_f64(&C(0, 3), c30);
    vst1q_f64(&C(2, 3), c31);
    vst1q_f64(&C(4, 3), c32);
    vst1q_f64(&C(6, 3), c33);
}

//------------------------------------------------------
// PackMatrixB_8_d - Copy a k×8 panel of B using NEON
//------------------------------------------------------
static void PackMatrixB_8_d(CBLAS_INDEX k, CBLAS_INDEX n_cols, double *b, CBLAS_INDEX ldb, double *b_to)
{
    if (n_cols >= NR_D) {
        // Fast path: full 8 columns, use NEON loads/stores
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            double *b_ij_pntr = &B(0, j);
            
            // Load and store 8 doubles using NEON (4 float64x2_t registers)
            float64x2_t b0 = vld1q_f64(b_ij_pntr);
            float64x2_t b1 = vld1q_f64(b_ij_pntr + 2);
            float64x2_t b2 = vld1q_f64(b_ij_pntr + 4);
            float64x2_t b3 = vld1q_f64(b_ij_pntr + 6);
            vst1q_f64(b_to, b0);
            vst1q_f64(b_to + 2, b1);
            vst1q_f64(b_to + 4, b2);
            vst1q_f64(b_to + 6, b3);
            
            b_to += NR_D;
        }
    } else {
        // Slow path: partial columns, need zero-padding
        for (CBLAS_INDEX j = 0; j < k; j++)
        {
            double *b_ij_pntr = &B(0, j);
            CBLAS_INDEX col;
            for (col = 0; col < n_cols; col++) {
                b_to[col] = b_ij_pntr[col];
            }
            for (; col < NR_D; col++) {
                b_to[col] = 0.0;
            }
            b_to += NR_D;
        }
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
// InnerKernel - NEON implementation with 4x8 micro-kernel for dgemm
// GotoBLAS-style: pack A once per row-block, pack B for each col-block
//------------------------------------------------------
static void InnerKernel_dgemm_neon(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                                   double* a, CBLAS_INDEX lda, 
                                   double* b, CBLAS_INDEX ldb, 
                                   double* c, CBLAS_INDEX ldc,
                                   double alpha, int thread_id)
{
    cblas_gemm_buffer_t* buf = cblas_get_gemm_buffer(thread_id);
    double* packedA;
    double* packedB;
    int use_pool = 0;
    
    if (buf) {
        packedA = buf->packedA_d;
        packedB = buf->packedB_d;
        use_pool = 1;
    } else {
        packedA = (double*)malloc(MR_D * k * sizeof(double));
        packedB = (double*)malloc(k * NR_D * sizeof(double));
        if (!packedA || !packedB) {
            free(packedA);
            free(packedB);
            return;
        }
    }

    CBLAS_INDEX row, col;

    // Main loop: 4 rows at a time
    for (row = 0; row + MR_D <= m; row += MR_D)
    {
        // Pack this 4×k panel of A once per row iteration
        PackMatrixA_4_d(k, MR_D, &A(0, row), lda, packedA);

        // Process 8 columns at a time with 4x8 micro-kernel
        for (col = 0; col + NR_D <= n; col += NR_D)
        {
            // Pack this k×8 panel of B
            PackMatrixB_8_d(k, NR_D, &B(col, 0), ldb, packedB);
            
            // Call 4x8 micro-kernel
            AddDot4x8_dgemm_neon(k, packedA, packedB, &C(col, row), ldc, alpha);
        }

        // Handle leftover columns (< 8)
        for (; col < n; col++) {
            for (CBLAS_INDEX r = 0; r < MR_D; r++) {
                AddDot_d(k, &A(0, row + r), 1, &B(col, 0), ldb, &C(col, row + r), alpha);
            }
        }
    }

    // Handle leftover rows (< 4)
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
// DGEMM kernel - NEON version (4x4 micro-kernel)
//------------------------------------------------------
void dgemm_k_neon(cblas_args_t* args)
{
    InnerKernel_dgemm_neon(args->ib, args->n, args->pb, 
                           (double*)args->a, args->lda, 
                           (double*)args->b, args->ldb, 
                           (double*)args->c, args->ldc,
                           args->alpha_d, args->thread_id);
}

#endif // ARM64 NEON
