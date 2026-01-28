//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

// Matrix sub-tile block sizes for caching data in contiguous memory
#define mc 256
#define kc 128
#define nb 1024

// macros to simpify matrix element access
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incx]

#define CHECK_GUARDS()  assert(aguard == 0xbaadf00d && bguard == 0xbaadf00d && cguard == 0xbaadf00d)

#ifdef USE_STATIC_BUFFERS
    static int aguard = 0xbaadf00d;
    static float packedA[mc * kc];
    static int bguard = 0xbaadf00d;
    static float packedB[kc * nb];
    static int cguard = 0xbaadf00d;
#endif

//------------------------------------------------------
// compute dot product of row of X and col of Y
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

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

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

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // load and duplicate 
        a_p0 = _mm_load_ps1(a);
        a_p1 = _mm_load_ps1(a + 1);
        a_p2 = _mm_load_ps1(a + 2);
        a_p3 = _mm_load_ps1(a + 3);

        a += 4;

        CHECK_ALIGN(b, 16);

        b_row = _mm_load_ps(b);

        b += 4;

        // rows 1 - 4 using FMADD
#ifdef USE_INTEL_FMA
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

#elif defined(__aarch64__) && defined(USE_SIMD)

static void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    float32x4_t c_row1, c_row2, c_row3, c_row4;
    float32x4_t b_row;
    float32x4_t a_p0, a_p1, a_p2, a_p3;
    
    // __builtin_prefetch(&C(0,0), 0);
    // __builtin_prefetch(&C(0,1), 0);
    // __builtin_prefetch(&C(0,2), 0);
    // __builtin_prefetch(&C(0,3), 0);
    // __builtin_prefetch(a, 0);
    // __builtin_prefetch(b, 0);

    // 4 x 4 floats into SIMD regs
    c_row1 = vld1q_f32(&C(0,0));
    c_row2 = vld1q_f32(&C(0,1));
    c_row3 = vld1q_f32(&C(0,2));
    c_row4 = vld1q_f32(&C(0,3));

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // load 1 float and duplicate to 4 SIMD elements 
        a_p0 = vld1q_dup_f32(a);
        a_p1 = vld1q_dup_f32(a + 1);
        a_p2 = vld1q_dup_f32(a + 2);
        a_p3 = vld1q_dup_f32(a + 3);

        a += 4;

        // load 4 floats
        b_row = vld1q_f32(b);

        b += 4;

#ifdef __ARM_FEATURE_FMA
        // rows 1 - 4 using NEON FMAD C += A * B
        c_row1 = vfmaq_f32(c_row1, a_p0, b_row);
        c_row2 = vfmaq_f32(c_row2, a_p1, b_row);
        c_row3 = vfmaq_f32(c_row3, a_p2, b_row);
        c_row4 = vfmaq_f32(c_row4, a_p3, b_row);

#else
        // rows 1 - 4 using NEON MUL and ADD C += A * B
        c_row1 = vaddq_f32(c_row1, vmulq_f32(a_p0, b_row));
        c_row2 = vaddq_f32(c_row2, vmulq_f32(a_p1, b_row));
        c_row3 = vaddq_f32(c_row3, vmulq_f32(a_p2, b_row));
        c_row4 = vaddq_f32(c_row4, vmulq_f32(a_p3, b_row));
#endif
    }

    // store 4 x 4 floats
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

    for (CBLAS_INDEX p = 0; p < k; p++) 
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
    for (CBLAS_INDEX p = 0; p < k; p += 4) 
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
// pack a sub-tile of B into contiguous memory
//------------------------------------------------------
static void PackMatrixB(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
{
    // loop over rows of B
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(0, j);

        *b_to       = *b_ij_pntr;
        *(b_to + 1) = *(b_ij_pntr + 1);
        *(b_to + 2) = *(b_ij_pntr + 2);
        *(b_to + 3) = *(b_ij_pntr + 3);

        b_to += 4;
    }
}

//------------------------------------------------------
// pack a sub-tile of A into contiguous memory
//------------------------------------------------------
static void PackMatrixA(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    CBLAS_INDEX i;
    float   *a_0i_pntr = &A(0,0), *a_1i_pntr = &A(0,1),
            *a_2i_pntr = &A(0,2), *a_3i_pntr = &A(0,3);

    // loop over cols of A
    for (i = 0; i < k; i++)
    {
        *a_to       = *a_0i_pntr++;
        *(a_to + 1) = *a_1i_pntr++;
        *(a_to + 2) = *a_2i_pntr++;
        *(a_to + 3) = *a_3i_pntr++;

        a_to += 4;
    }
}

//------------------------------------------------------
// GEMM kernel
//------------------------------------------------------
static void InnerKernel(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc)
{
#if !defined(USE_STATIC_BUFFERS)
    #ifdef _WIN32
        int aguard = 0xbaadf00d;
        float* packedA = _malloca(mc * kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = _malloca(kc * nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #else
        int aguard = 0xbaadf00d;
        float* packedA = alloca(mc * kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = alloca(kc * nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #endif
#endif
    CHECK_GUARDS();

//printf("tile: (%ld, %ld) x (%ld, %ld)\n", k, m, n, k);

    //int row_leftover    = m % 4; 
    //int col_leftover    = n % 4;
    CBLAS_INDEX row, col;

    // Loop over the rows and columns of C unrolled by 4
    for (row = 0; row + 4 <= m; row += 4)
    {
        // we are pre-caching all cols so we need to do it only once
        if (row == 0)
            PackMatrixB(k, &B(0, row), ldb, &packedB[row * k]);

        for (col = 0; col + 4 <= n; col += 4)
        {
            // we are pre-caching all rows so we need to do it only once
            if (col == 0) 
                PackMatrixA(k, &A(col, 0), lda, &packedA[row * k]);

            AddDot4x4(k, &packedA[row * k], 4, &packedB[col * k], k, &C(col, row), ldc);
        }

        // handle leftover columns
        switch(n - col)
        {
            case 3:     
                AddDot(k, &A(0, row), 1, &B(col + 2, 0), ldb, &C(col + 2, row));
                AddDot(k, &A(0, row+1), 1, &B(col + 2, 0), ldb, &C(col + 2, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col + 2, 0), ldb, &C(col + 2, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col + 2, 0), ldb, &C(col + 2, row+3));
            case 2:
                AddDot(k, &A(0, row), 1, &B(col + 1, 0), ldb, &C(col + 1, row));
                AddDot(k, &A(0, row+1), 1, &B(col + 1, 0), ldb, &C(col + 1, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col + 1, 0), ldb, &C(col + 1, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col + 1, 0), ldb, &C(col + 1, row+3));
            case 1:
                //AddDot1x4(k, &A(0, row), lda, &B(col, 0), ldb, &C(col, row), ldc);
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
                AddDot(k, &A(0, row+1), 1, &B(col, 0), ldb, &C(col, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col, 0), ldb, &C(col, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col, 0), ldb, &C(col, row+3));
            case 0: ;   // nothing to do!
        }
    }

    // handle leftover rows    
    switch(m - row)
    {
        case 3:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 2), 1, &B(col, 0), ldb, &C(col, row + 2));
        case 2:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1));
        case 1:    for (col = 0; col < n; col++) AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
        case 0: ;   // nothing to do!
    }

    CHECK_GUARDS();
}

//------------------------------------------------------
//
//------------------------------------------------------
void sgemm_k(cblas_args_t* args)
{
    InnerKernel(args->ib, args->n, args->pb, args->a, args->lda, args->b, args->ldb, args->c, args->ldc);
}

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
#ifdef CBLAS_CHECK_INPUTS
    CBLAS_INDEX nota = (transa == CblasNoTrans);
    CBLAS_INDEX notb = (transb == CblasNoTrans);
    CBLAS_INDEX nrowc = m;

    if (layout == CblasRowMajor)
    {
        nota = !nota;
        notb = !notb;
        nrowc = n;
    }

    CBLAS_INDEX nrowa, nrowb;

    if (nota)
        nrowa = m;
    else
        nrowa = k;

    if (notb)
        nrowb = k;
    else
        nrowb = n;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (m < 0)
        info = 4;
    else if (n < 0)
        info = 5;
    else if (k < 0)
        info = 6;
    else if (!a)
        info = 8;
    else if (lda < MAX(1, nrowa))
        info = 9;
    else if (!b)
        info = 10;
    else if (ldb < MAX(1, nrowb))
        info = 11;
    else if (!c)
        info = 12;
    else if (ldc < MAX(1, nrowc))
        info = 13;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (m < 0 || n < 0 || k < 0 || !a || !b || !c || lda < MAX(1, nrowa) || ldb < MAX(1, nrowb) || ldc < MAX(1, m))
    {
        assert(layout == CblasRowMajor || layout == CblasColMajor);
        assert(transa == CblasTrans || transa == CblasNoTrans);
        assert(transb == CblasTrans || transb == CblasNoTrans);
        assert(m > 0 && n > 0 && k > 0);
        assert(a && b && c);
        assert(alpha != 0.0f && beta != 0.0f);
        return;
    }
#endif
#endif

    CBLAS_INDEX pb, ib;

#if defined(MT_ENABLED)
    CBLAS_INDEX horiz_tiles = k / kc + 1;
    CBLAS_INDEX vert_tiles = m / mc + 1;
    CBLAS_INDEX total_tiles = horiz_tiles * vert_tiles;
    int tile_count = 0;

    //printf("tile count = %u\n", total_tiles);
    #ifdef _WIN32
        work_queue_t *queue = _malloca(total_tiles * sizeof(work_queue_t));
        cblas_args_t *args = _malloca(total_tiles * sizeof(cblas_args_t));
    #else
        work_queue_t *queue = alloca(total_tiles * sizeof(work_queue_t));
        cblas_args_t *args = alloca(total_tiles * sizeof(cblas_args_t));
    #endif

    // Compute an mc x n block of C by a call to the InnerKernel
    for (int p = 0; p < k; p += kc) 
    {
        pb = MIN(k - p, kc);
        for (int row = 0; row < m; row += mc) 
        {
            ib = MIN(m - row, mc);

            args[tile_count].incx = 1;
            args[tile_count].incy = 1;
            args[tile_count].n = n;
            args[tile_count].lda = lda;
            args[tile_count].ldb = ldb;
            args[tile_count].ldc = ldc;
            args[tile_count].a = &A(p, row);
            args[tile_count].b = &B(0, p);
            args[tile_count].c = &C(0, row);
            args[tile_count].ib = ib;
            args[tile_count].pb = pb;
        
            queue[tile_count].finished   = 0;
            queue[tile_count].args       = &args[tile_count];
            queue[tile_count].kernel     = blas_kernels.sgemm_k;
            queue[tile_count].next       = &queue[tile_count + 1];

            tile_count++;
            // InnerKernel(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
        }
    }

    assert(tile_count <= total_tiles);

    // mark end of task queue
    queue[tile_count - 1].next = NULL;

    // synchronously execute task queue
    cblas_execute(tile_count, queue);

#else
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
#endif
}

//------------------------------------------------------
// single-precision reference matrix multipl
//------------------------------------------------------
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
    assert(layout == CblasRowMajor || layout == CblasColMajor);
    assert(transa == CblasTrans || transa == CblasNoTrans);
    assert(transb == CblasTrans || transb == CblasNoTrans);
    assert(m > 0 && n > 0 && k > 0);
    assert(a && b && c);
    assert(alpha != 0.0f && beta != 0.0f);

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
#ifdef CBLAS_CHECK_INPUTS
    CBLAS_INDEX nota = (transa == CblasNoTrans);
    CBLAS_INDEX notb = (transb == CblasNoTrans);
    CBLAS_INDEX nrowc = m;

    if (layout == CblasRowMajor)
    {
        nota = !nota;
        notb = !notb;
        nrowc = n;
    }

    CBLAS_INDEX nrowa, nrowb;

    if (nota)
        nrowa = m;
    else
        nrowa = k;

    if (notb)
        nrowb = k;
    else
        nrowb = n;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (m < 0)
        info = 4;
    else if (n < 0)
        info = 5;
    else if (k < 0)
        info = 6;
    else if (!a)
        info = 8;
    else if (lda < MAX(1, nrowa))
        info = 9;
    else if (!b)
        info = 10;
    else if (ldb < MAX(1, nrowb))
        info = 11;
    else if (!c)
        info = 12;
    else if (ldc < MAX(1, nrowc))
        info = 13;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (m < 0 || n < 0 || k < 0 || !a || !b || !c || lda < MAX(1, nrowa) || ldb < MAX(1, nrowb) || ldc < MAX(1, m))
    {
        assert(layout == CblasRowMajor || layout == CblasColMajor);
        assert(transa == CblasTrans || transa == CblasNoTrans);
        assert(transb == CblasTrans || transb == CblasNoTrans);
        assert(m > 0 && n > 0 && k > 0);
        assert(a && b && c);
        assert(alpha != 0.0 && beta != 0.0);
        return;
    }
#endif
#endif

    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            for (int p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

