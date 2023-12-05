//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include <stdio.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

// Block sizes
#define mc 256
#define kc 128

// macros to simpify matrix element access
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incx]

static float packedA[mc * kc];

//------------------------------------------------------
// compute dot product of row of X and col of Y
//------------------------------------------------------
static void AddDot(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, float *gamma)
{
	for (int p = 0; p < k; p++)
    {
		*gamma += x[p] * Y(p);
	}
}

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows
//------------------------------------------------------
static void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    __m128 c_row1, c_row2, c_row3, c_row4;
    __m128 b_row;
    __m128 a_p0, a_p1, a_p2, a_p3;
    
    c_row1 = _mm_load_ps(&C(0,0));
    c_row2 = _mm_load_ps(&C(0,1));
    c_row3 = _mm_load_ps(&C(0,2));
    c_row4 = _mm_load_ps(&C(0,3));

	for (int p = 0; p < k; p++) 
    {
        // load and duplicate 
        a_p0 = _mm_load_ps1(a);
        a_p1 = _mm_load_ps1(a + 1);
        a_p2 = _mm_load_ps1(a + 2);
        a_p3 = _mm_load_ps1(a + 3);

        a += 4;

        b_row = _mm_load_ps(&B(0, p));

        // rows 1 - 4 using FMADD
#if 0
        c_row1 = _mm_fmadd_ps(a_p0, b_row, c_row1);
        c_row2 = _mm_fmadd_ps(a_p1, b_row, c_row2);
        c_row3 = _mm_fmadd_ps(a_p2, b_row, c_row3);
        c_row4 = _mm_fmadd_ps(a_p3, b_row, c_row4);
#else
        // rows 1 - 4 using SSE3
        c_row1 = _mm_add_ps(c_row1, _mm_mul_ps(a_p0, b_row));
        c_row2 = _mm_add_ps(c_row2, _mm_mul_ps(a_p1, b_row));
        c_row3 = _mm_add_ps(c_row3, _mm_mul_ps(a_p2, b_row));
        c_row4 = _mm_add_ps(c_row4, _mm_mul_ps(a_p3, b_row));
#endif
    }

    _mm_store_ps(&C(0, 0), c_row1);
    _mm_store_ps(&C(0, 1), c_row2);
    _mm_store_ps(&C(0, 2), c_row3);
    _mm_store_ps(&C(0, 3), c_row4);
}

#elif defined(__aarch64__)

static void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    float32x4_t c_row1, c_row2, c_row3, c_row4;
    float32x4_t b_row;
    float32x4_t a_p0, a_p1, a_p2, a_p3;
    
    c_row1 = vld1q_f32(&C(0,0));
    c_row2 = vld1q_f32(&C(0,1));
    c_row3 = vld1q_f32(&C(0,2));
    c_row4 = vld1q_f32(&C(0,3));

	for (int p = 0; p < k; p++) 
    {
        a_p0 = vld1q_dup_f32(a);
        a_p1 = vld1q_dup_f32(a + 1);
        a_p2 = vld1q_dup_f32(a + 2);
        a_p3 = vld1q_dup_f32(a + 3);

        a += 4;

        b_row = vld1q_f32(&B(0, p));

#ifdef __ARM_FEATURE_FMA
        // rows 1 - 4 using NEON FMAD
        c_row1 = vfmaq_f32(c_row1, a_p0, b_row);
        c_row2 = vfmaq_f32(c_row2, a_p1, b_row);
        c_row3 = vfmaq_f32(c_row3, a_p2, b_row);
        c_row4 = vfmaq_f32(c_row4, a_p3, b_row);

#else
        // rows 1 - 4 using NEON
        c_row1 = vaddq_f32(c_row1, vmulq_f32(a_p0, b_row));
        c_row2 = vaddq_f32(c_row2, vmulq_f32(a_p1, b_row));
        c_row3 = vaddq_f32(c_row3, vmulq_f32(a_p2, b_row));
        c_row4 = vaddq_f32(c_row4, vmulq_f32(a_p3, b_row));
#endif
    }

    vst1q_f32(&C(0, 0), c_row1);
    vst1q_f32(&C(0, 1), c_row2);
    vst1q_f32(&C(0, 2), c_row3);
    vst1q_f32(&C(0, 3), c_row4);
}

#else   // fall-back non-vector version

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows
//------------------------------------------------------
static void AddDot4x4(CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc)
{
    register float 
        c_00, c_10, c_20, c_30,
        c_01, c_11, c_21, c_31,
        c_02, c_12, c_22, c_32,
        c_03, c_13, c_23, c_33;

    register float a_p0, a_p1, a_p2, a_p3;
    register float b_0p, b_1p, b_2p, b_3p;
    float* a_p0_ptr, * a_p1_ptr, * a_p2_ptr, * a_p3_ptr;

    a_p0_ptr = &A(0, 0);
    a_p1_ptr = &A(0, 1);
    a_p2_ptr = &A(0, 2);
    a_p3_ptr = &A(0, 3);

    c_00 = 0.0f; c_10 = 0.0f; c_20 = 0.0f; c_30 = 0.0f;
    c_01 = 0.0f; c_11 = 0.0f; c_21 = 0.0f; c_31 = 0.0f;
    c_02 = 0.0f; c_12 = 0.0f; c_22 = 0.0f; c_32 = 0.0f;
    c_03 = 0.0f; c_13 = 0.0f; c_23 = 0.0f; c_33 = 0.0f;

    for (int p = 0; p < k; p++) 
    {
        a_p0 = *a;
        a_p1 = *(a + 1);
        a_p2 = *(a + 2);
        a_p3 = *(a + 3);

        a += 4;

        b_0p = B(0, p);
        b_1p = B(1, p);
        b_2p = B(2, p);
        b_3p = B(3, p);

        // row 1
        c_00 += a_p0 * b_0p;
        c_10 += a_p0 * b_1p;
        c_20 += a_p0 * b_2p;
        c_30 += a_p0 * b_3p;

        // row 2
        c_01 += a_p1 * b_0p;
        c_11 += a_p1 * b_1p;
        c_21 += a_p1 * b_2p;
        c_31 += a_p1 * b_3p;

        // row 3
        c_02 += a_p2 * b_0p;
        c_12 += a_p2 * b_1p;
        c_22 += a_p2 * b_2p;
        c_32 += a_p2 * b_3p;

        // row 4
        c_03 += a_p3 * b_0p;
        c_13 += a_p3 * b_1p;
        c_23 += a_p3 * b_2p;
        c_33 += a_p3 * b_3p;
    }

    C(0, 0) += c_00; C(1, 0) += c_10; C(2, 0) += c_20; C(3,0) += c_30;
    C(0, 1) += c_01; C(1, 1) += c_11; C(2, 1) += c_21; C(3,1) += c_31;
    C(0, 2) += c_02; C(1, 2) += c_12; C(2, 2) += c_22; C(3,2) += c_32;
    C(0, 3) += c_03; C(1, 3) += c_13; C(2, 3) += c_23; C(3,3) += c_33;
}

#endif

//------------------------------------------------------
// compute 4 dot products at a time
// 4 rows of A by 1 column of B
//------------------------------------------------------
static void AddDot1x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    register float c_00, c_01, c_02, c_03, b_0p;
    float *a0, *a1, *a2, *a3;

    // grab the start of 4 rows of A
    a0 = &A(0, 0);
    a1 = &A(0, 1);
    a2 = &A(0, 2);
    a3 = &A(0, 3);

    c_00 = 0.0f;
    c_01 = 0.0f;
    c_02 = 0.0f;
    c_03 = 0.0f;

    // 
    for (int p = 0; p < k; p += 4) 
    {
        b_0p = B(0, p);

        c_00 += *a0 * b_0p;
        c_01 += *a1 * b_0p;
        c_02 += *a2 * b_0p;
        c_03 += *a3 * b_0p;

        b_0p = B(0, p + 1);

        c_00 += *(a0+1) * b_0p;
        c_01 += *(a1+1) * b_0p;
        c_02 += *(a2+1) * b_0p;
        c_03 += *(a3+1) * b_0p;

        b_0p = B(0, p + 2);

        c_00 += *(a0+2) * b_0p;
        c_01 += *(a1+2) * b_0p;
        c_02 += *(a2+2) * b_0p;
        c_03 += *(a3+2) * b_0p;

        b_0p = B(0, p + 3);

        c_00 += *(a0+3) * b_0p;
        c_01 += *(a1+3) * b_0p;
        c_02 += *(a2+3) * b_0p;
        c_03 += *(a3+3) * b_0p;

        a0 += 4;
        a1 += 4;
        a2 += 4;
        a3 += 4;
    }

    C(0,0) += c_00;
    C(0,1) += c_01;
    C(0,2) += c_02;
    C(0,3) += c_03;
}

//------------------------------------------------------
//
//------------------------------------------------------
static void PackMatrixB(int k, float *b, int ldb, float* b_to)
{
    /* loop over cols of B */
    for (int j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(j, 0);

        *b_to++ = *b_ij_pntr;
        *b_to++ = *(b_ij_pntr + 1);
        *b_to++ = *(b_ij_pntr + 2);
        *b_to++ = *(b_ij_pntr + 3);
    }
}

//------------------------------------------------------
// pack a sub-tile of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA(int k, float *a, int lda, float *a_to)
{
    int i;
    float   *a_0i_pntr = &A(0,0), *a_1i_pntr = &A(0,1),
            *a_2i_pntr = &A(0,2), *a_3i_pntr = &A(0,3);

    for (i = 0; i < k; i++) {  /* loop over rows of A */
        *a_to++ = *a_0i_pntr++;
        *a_to++ = *a_1i_pntr++;
        *a_to++ = *a_2i_pntr++;
        *a_to++ = *a_3i_pntr++;
    }
}

//------------------------------------------------------
//
//------------------------------------------------------
static void InnerKernel(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc)
{
//    float packedA[m * k];

    //int row_leftover    = m % 4; 
    //int col_leftover    = n % 4;
    int row, col;

    // Loop over the rows and columns of C unrolled by 4
    for (row = 0; row < m; row += 4)
    {
        for (col = 0; col < n; col += 4)
        {
            //AddDot4x4(k, &A(0, row), lda, &B(col, 0), ldb, &C(col, row), ldc);

            if (row == 0) PackMatrixA(k, &A(col, 0), lda, &packedA[row * k]);
            AddDot4x4(k, &packedA[row * k], 4, &B(0, row), ldb, &C(col, row), ldc);
        }

        // use Duff's device to handle leftover columns
        // printf("col_leftover = %d, col = %d\n", col_leftover, col);
        //switch(col_leftover)
        //{
        //    case 3:     AddDot(k, &A(0, row), lda, &B(col + 2, 0), &C(col + 2, row));
        //    case 2:     AddDot(k, &A(0, row), lda, &B(col + 1, 0), &C(col + 1, row));
        //    case 1:     AddDot(k, &A(0, row), lda, &B(col, 0), &C(col, row));
        //    case 0: ;   // nothing to do!
        //}
    }

    // switch(row_leftover)
    // {
    //     case 3:
    //     case 2:
    //     case 1:
    //     case 0: ;   // nothing to do!
    // }
}

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float beta, float *c, CBLAS_INDEX ldc)
{
    CBLAS_INDEX pb, ib;

    // Compute an mc x n block of C by a call to the InnerKernel
    for (int p = 0; p < k; p += kc) 
    {
        pb = MIN(k - p, kc);
        for (int row = 0; row < m; row += mc) 
        {
            ib = MIN(m - row, mc);
            InnerKernel(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
        }
    }

    // TODO - handle remainder for matrices that are not multiples of 4 in size!
}

//------------------------------------------------------
// single-precision reference matrix multipl
//------------------------------------------------------
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

//------------------------------------------------------
// double-precision general matrix multiply
//------------------------------------------------------
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc)
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

