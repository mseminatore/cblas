//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - NEON implementation for ARM64
//------------------------------------------------------

#include "cblas.h"

#if defined(__aarch64__) && defined(__ARM_NEON)

#include "cblas_simd.h"
#include <stdlib.h>

// Prefetch distance tuning
#define PREFETCH_DISTANCE 16

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// compute dot product of row of X and col of Y (scalar fallback)
//------------------------------------------------------
static void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *gamma)
{
    float *px = x;
    float *py = y;
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        *gamma += (*px) * (*py);
        px += incx;
        py += incy;
    }
}

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows (NEON)
//------------------------------------------------------
static void AddDot4x4_neon(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    float32x4_t c_row1, c_row2, c_row3, c_row4;
    float32x4_t b_row;
    float32x4_t a_p0, a_p1, a_p2, a_p3;
    
    // Load 4 x 4 floats into SIMD regs
    c_row1 = vld1q_f32(&C(0,0));
    c_row2 = vld1q_f32(&C(0,1));
    c_row3 = vld1q_f32(&C(0,2));
    c_row4 = vld1q_f32(&C(0,3));

    for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // Load 1 float and duplicate to 4 SIMD elements 
        a_p0 = vld1q_dup_f32(a);
        a_p1 = vld1q_dup_f32(a + 1);
        a_p2 = vld1q_dup_f32(a + 2);
        a_p3 = vld1q_dup_f32(a + 3);

        // Prefetch data ahead
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * 4), 0, 3);
        }

        a += 4;

        // Load 4 floats
        b_row = vld1q_f32(b);

        b += 4;

#ifdef __ARM_FEATURE_FMA
        // Rows 1-4 using NEON FMA: C += A * B
        c_row1 = vfmaq_f32(c_row1, a_p0, b_row);
        c_row2 = vfmaq_f32(c_row2, a_p1, b_row);
        c_row3 = vfmaq_f32(c_row3, a_p2, b_row);
        c_row4 = vfmaq_f32(c_row4, a_p3, b_row);
#else
        // Rows 1-4 using NEON MUL and ADD: C += A * B
        c_row1 = vaddq_f32(c_row1, vmulq_f32(a_p0, b_row));
        c_row2 = vaddq_f32(c_row2, vmulq_f32(a_p1, b_row));
        c_row3 = vaddq_f32(c_row3, vmulq_f32(a_p2, b_row));
        c_row4 = vaddq_f32(c_row4, vmulq_f32(a_p3, b_row));
#endif
    }

    // Store 4 x 4 floats
    vst1q_f32(&C(0, 0), c_row1);
    vst1q_f32(&C(0, 1), c_row2);
    vst1q_f32(&C(0, 2), c_row3);
    vst1q_f32(&C(0, 3), c_row4);
}

//------------------------------------------------------
// PackMatrixB - Copy a k×4 panel of B into contiguous memory
//------------------------------------------------------
static void PackMatrixB(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
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
// PackMatrixA - Copy a 4×k panel of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
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
// InnerKernel - NEON implementation
//------------------------------------------------------
static void InnerKernel_neon(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, 
                             float* a, CBLAS_INDEX lda, 
                             float* b, CBLAS_INDEX ldb, 
                             float* c, CBLAS_INDEX ldc)
{
    float* packedA = (float*)malloc(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
    float* packedB = (float*)malloc(cblas_gemm_kc * cblas_gemm_nb * sizeof(float));
    
    if (!packedA || !packedB) {
        free(packedA);
        free(packedB);
        return;
    }

    CBLAS_INDEX row, col;

    for (row = 0; row + 4 <= m; row += 4)
    {
        if (row == 0)
            PackMatrixB(k, &B(0, 0), ldb, packedB);

        for (col = 0; col + 4 <= n; col += 4)
        {
            if (col == 0) 
                PackMatrixA(k, &A(0, row), lda, packedA);

            AddDot4x4_neon(k, packedA, 4, packedB, k, &C(col, row), ldc);
        }

        // handle leftover columns
        switch(n - col)
        {
            case 3:     
                AddDot(k, &A(0, row), 1, &B(col + 2, 0), ldb, &C(col + 2, row));
                AddDot(k, &A(0, row+1), 1, &B(col + 2, 0), ldb, &C(col + 2, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col + 2, 0), ldb, &C(col + 2, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col + 2, 0), ldb, &C(col + 2, row+3));
                CBLAS_FALLTHROUGH;
            case 2:
                AddDot(k, &A(0, row), 1, &B(col + 1, 0), ldb, &C(col + 1, row));
                AddDot(k, &A(0, row+1), 1, &B(col + 1, 0), ldb, &C(col + 1, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col + 1, 0), ldb, &C(col + 1, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col + 1, 0), ldb, &C(col + 1, row+3));
                CBLAS_FALLTHROUGH;
            case 1:
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
                AddDot(k, &A(0, row+1), 1, &B(col, 0), ldb, &C(col, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col, 0), ldb, &C(col, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col, 0), ldb, &C(col, row+3));
                CBLAS_FALLTHROUGH;
            case 0: ;
        }
    }

    // handle leftover rows
    switch(m - row)
    {
        case 3: for (col = 0; col < n; col++) AddDot(k, &A(0, row + 2), 1, &B(col, 0), ldb, &C(col, row + 2));
            CBLAS_FALLTHROUGH;
        case 2: for (col = 0; col < n; col++) AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1));
            CBLAS_FALLTHROUGH;
        case 1: for (col = 0; col < n; col++) AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
            CBLAS_FALLTHROUGH;
        case 0: ;
    }
    
    free(packedA);
    free(packedB);
}

//------------------------------------------------------
// SGEMM kernel - NEON version
//------------------------------------------------------
void sgemm_k_neon(cblas_args_t* args)
{
    InnerKernel_neon(args->ib, args->n, args->pb, 
                     args->a, args->lda, 
                     args->b, args->ldb, 
                     args->c, args->ldc);
}

#endif // ARM64 NEON
