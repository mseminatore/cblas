//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
// GEMM kernel - AVX-512 implementation for x86-64
// Implements 6x32 micro-kernel for optimal AVX-512 register utilization
// Each ZMM register holds 16 floats, so 32 columns = 2 ZMM
// 6 rows × 2 ZMM = 12 ZMM for C accumulator
// Plus 2 ZMM for B, 1 ZMM for A broadcast = 15 ZMM total
//------------------------------------------------------

#include "cblas.h"

#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX512F__)

#include "cblas_simd.h"
#include <stdlib.h>

// Micro-kernel dimensions for AVX-512
#define MR_512 6   // Rows per micro-kernel
#define NR_512 32  // Columns per micro-kernel (2 ZMM registers of 16 floats)

// Matrix access macros (local to this file)
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//------------------------------------------------------
// Scalar fallback for edge cases
//------------------------------------------------------
static void AddDot_avx512(CBLAS_INDEX k, float *x, CBLAS_INDEX incx, float *y, CBLAS_INDEX incy, float *gamma, float alpha)
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
// 6x32 micro-kernel using AVX-512
// Computes C[6x32] += alpha * A[6xk] * B[kx32]
// Uses 12 ZMM registers for C (6 rows × 2 ZMM per row = 32 cols)
// 2 ZMM for B loads, broadcasts from A
// K-loop unrolled by 4 for better ILP
//------------------------------------------------------
static void AddDot6x32_avx512(CBLAS_INDEX k, float *a, float *b, float *c, CBLAS_INDEX ldc, float alpha)
{
    // C accumulator registers: 6 rows × 2 ZMM (16 floats each) = 32 columns
    __m512 c00, c01;  // Row 0: columns 0-15, 16-31
    __m512 c10, c11;  // Row 1
    __m512 c20, c21;  // Row 2
    __m512 c30, c31;  // Row 3
    __m512 c40, c41;  // Row 4
    __m512 c50, c51;  // Row 5
    
    __m512 b0, b1;    // B row: 32 floats = 2 ZMM
    __m512 a_elem;    // A element broadcast
    
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    
    // Initialize accumulators to zero
    c00 = _mm512_setzero_ps(); c01 = _mm512_setzero_ps();
    c10 = _mm512_setzero_ps(); c11 = _mm512_setzero_ps();
    c20 = _mm512_setzero_ps(); c21 = _mm512_setzero_ps();
    c30 = _mm512_setzero_ps(); c31 = _mm512_setzero_ps();
    c40 = _mm512_setzero_ps(); c41 = _mm512_setzero_ps();
    c50 = _mm512_setzero_ps(); c51 = _mm512_setzero_ps();
    
    CBLAS_INDEX p = 0;
    
    // Main loop: unrolled by 2 for AVX-512
    for (; p + 2 <= k; p += 2)
    {
        // Iteration 0
        b0 = _mm512_loadu_ps(b);
        b1 = _mm512_loadu_ps(b + 16);
        
        a_elem = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(a_elem, b0, c00);
        c01 = _mm512_fmadd_ps(a_elem, b1, c01);
        
        a_elem = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(a_elem, b0, c10);
        c11 = _mm512_fmadd_ps(a_elem, b1, c11);
        
        a_elem = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(a_elem, b0, c20);
        c21 = _mm512_fmadd_ps(a_elem, b1, c21);
        
        a_elem = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(a_elem, b0, c30);
        c31 = _mm512_fmadd_ps(a_elem, b1, c31);
        
        a_elem = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(a_elem, b0, c40);
        c41 = _mm512_fmadd_ps(a_elem, b1, c41);
        
        a_elem = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(a_elem, b0, c50);
        c51 = _mm512_fmadd_ps(a_elem, b1, c51);
        
        // Iteration 1
        b0 = _mm512_loadu_ps(b + NR_512);
        b1 = _mm512_loadu_ps(b + NR_512 + 16);
        
        a_elem = _mm512_set1_ps(a[MR_512]);
        c00 = _mm512_fmadd_ps(a_elem, b0, c00);
        c01 = _mm512_fmadd_ps(a_elem, b1, c01);
        
        a_elem = _mm512_set1_ps(a[MR_512 + 1]);
        c10 = _mm512_fmadd_ps(a_elem, b0, c10);
        c11 = _mm512_fmadd_ps(a_elem, b1, c11);
        
        a_elem = _mm512_set1_ps(a[MR_512 + 2]);
        c20 = _mm512_fmadd_ps(a_elem, b0, c20);
        c21 = _mm512_fmadd_ps(a_elem, b1, c21);
        
        a_elem = _mm512_set1_ps(a[MR_512 + 3]);
        c30 = _mm512_fmadd_ps(a_elem, b0, c30);
        c31 = _mm512_fmadd_ps(a_elem, b1, c31);
        
        a_elem = _mm512_set1_ps(a[MR_512 + 4]);
        c40 = _mm512_fmadd_ps(a_elem, b0, c40);
        c41 = _mm512_fmadd_ps(a_elem, b1, c41);
        
        a_elem = _mm512_set1_ps(a[MR_512 + 5]);
        c50 = _mm512_fmadd_ps(a_elem, b0, c50);
        c51 = _mm512_fmadd_ps(a_elem, b1, c51);
        
        a += 2 * MR_512;
        b += 2 * NR_512;
    }
    
    // Handle remaining k iteration
    for (; p < k; p++)
    {
        b0 = _mm512_loadu_ps(b);
        b1 = _mm512_loadu_ps(b + 16);
        
        a_elem = _mm512_set1_ps(a[0]);
        c00 = _mm512_fmadd_ps(a_elem, b0, c00);
        c01 = _mm512_fmadd_ps(a_elem, b1, c01);
        
        a_elem = _mm512_set1_ps(a[1]);
        c10 = _mm512_fmadd_ps(a_elem, b0, c10);
        c11 = _mm512_fmadd_ps(a_elem, b1, c11);
        
        a_elem = _mm512_set1_ps(a[2]);
        c20 = _mm512_fmadd_ps(a_elem, b0, c20);
        c21 = _mm512_fmadd_ps(a_elem, b1, c21);
        
        a_elem = _mm512_set1_ps(a[3]);
        c30 = _mm512_fmadd_ps(a_elem, b0, c30);
        c31 = _mm512_fmadd_ps(a_elem, b1, c31);
        
        a_elem = _mm512_set1_ps(a[4]);
        c40 = _mm512_fmadd_ps(a_elem, b0, c40);
        c41 = _mm512_fmadd_ps(a_elem, b1, c41);
        
        a_elem = _mm512_set1_ps(a[5]);
        c50 = _mm512_fmadd_ps(a_elem, b0, c50);
        c51 = _mm512_fmadd_ps(a_elem, b1, c51);
        
        a += MR_512;
        b += NR_512;
    }
    
    // Apply alpha and accumulate to C
    __m512 c_load;
    
    c_load = _mm512_loadu_ps(c);
    c_load = _mm512_fmadd_ps(alpha_vec, c00, c_load);
    _mm512_storeu_ps(c, c_load);
    c_load = _mm512_loadu_ps(c + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c01, c_load);
    _mm512_storeu_ps(c + 16, c_load);
    
    c_load = _mm512_loadu_ps(c + ldc);
    c_load = _mm512_fmadd_ps(alpha_vec, c10, c_load);
    _mm512_storeu_ps(c + ldc, c_load);
    c_load = _mm512_loadu_ps(c + ldc + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c11, c_load);
    _mm512_storeu_ps(c + ldc + 16, c_load);
    
    c_load = _mm512_loadu_ps(c + 2*ldc);
    c_load = _mm512_fmadd_ps(alpha_vec, c20, c_load);
    _mm512_storeu_ps(c + 2*ldc, c_load);
    c_load = _mm512_loadu_ps(c + 2*ldc + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c21, c_load);
    _mm512_storeu_ps(c + 2*ldc + 16, c_load);
    
    c_load = _mm512_loadu_ps(c + 3*ldc);
    c_load = _mm512_fmadd_ps(alpha_vec, c30, c_load);
    _mm512_storeu_ps(c + 3*ldc, c_load);
    c_load = _mm512_loadu_ps(c + 3*ldc + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c31, c_load);
    _mm512_storeu_ps(c + 3*ldc + 16, c_load);
    
    c_load = _mm512_loadu_ps(c + 4*ldc);
    c_load = _mm512_fmadd_ps(alpha_vec, c40, c_load);
    _mm512_storeu_ps(c + 4*ldc, c_load);
    c_load = _mm512_loadu_ps(c + 4*ldc + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c41, c_load);
    _mm512_storeu_ps(c + 4*ldc + 16, c_load);
    
    c_load = _mm512_loadu_ps(c + 5*ldc);
    c_load = _mm512_fmadd_ps(alpha_vec, c50, c_load);
    _mm512_storeu_ps(c + 5*ldc, c_load);
    c_load = _mm512_loadu_ps(c + 5*ldc + 16);
    c_load = _mm512_fmadd_ps(alpha_vec, c51, c_load);
    _mm512_storeu_ps(c + 5*ldc + 16, c_load);
}

//------------------------------------------------------
// Pack A panel into contiguous memory (MR_512 × kc)
//------------------------------------------------------
static void PackMatrixA_avx512(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *packedA)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *a_ptr = &A(j, 0);
        *packedA++ = *a_ptr;
        *packedA++ = *(a_ptr + lda);
        *packedA++ = *(a_ptr + 2*lda);
        *packedA++ = *(a_ptr + 3*lda);
        *packedA++ = *(a_ptr + 4*lda);
        *packedA++ = *(a_ptr + 5*lda);
    }
}

//------------------------------------------------------
// Pack B panel into contiguous memory (kc × NR_512)
// Use AVX-512 for vectorized packing
//------------------------------------------------------
static void PackMatrixB_avx512(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *packedB)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        float *b_ptr = &B(0, i);
        __m512 b0 = _mm512_loadu_ps(b_ptr);
        __m512 b1 = _mm512_loadu_ps(b_ptr + 16);
        _mm512_storeu_ps(packedB, b0);
        _mm512_storeu_ps(packedB + 16, b1);
        packedB += NR_512;
    }
}

//------------------------------------------------------
// Inner kernel: multiply mc×kc panel of A by kc×nc panel of B
//------------------------------------------------------
static void InnerKernel_avx512(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                               float *a, CBLAS_INDEX lda,
                               float *b, CBLAS_INDEX ldb,
                               float *c, CBLAS_INDEX ldc,
                               float alpha, int first_time,
                               float *packedA, float *packedB)
{
    CBLAS_INDEX i, j;
    
    // Pack B on first use
    for (j = 0; j + NR_512 <= n; j += NR_512)
    {
        if (first_time)
            PackMatrixB_avx512(k, &B(j, 0), ldb, &packedB[j * k]);
    }
    
    for (i = 0; i + MR_512 <= m; i += MR_512)
    {
        PackMatrixA_avx512(k, &A(0, i), lda, packedA);
        
        for (j = 0; j + NR_512 <= n; j += NR_512)
        {
            AddDot6x32_avx512(k, packedA, &packedB[j * k], &C(j, i), ldc, alpha);
        }
        
        // Handle remaining columns with scalar fallback
        for (; j < n; j++)
        {
            for (CBLAS_INDEX ii = 0; ii < MR_512; ii++)
            {
                AddDot_avx512(k, packedA + ii, MR_512, &B(j, 0), ldb, &C(j, i + ii), alpha);
            }
        }
    }
    
    // Handle remaining rows with scalar fallback
    for (; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            AddDot_avx512(k, &A(0, i), 1, &B(j, 0), ldb, &C(j, i), alpha);
        }
    }
}

//------------------------------------------------------
// SGEMM AVX-512 kernel entry point
//------------------------------------------------------
void sgemm_k_avx512(cblas_args_t* args)
{
    float *a = (float*)args->a;
    float *b = (float*)args->b;
    float *c = (float*)args->c;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX k = args->k;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX ldb = args->ldb;
    CBLAS_INDEX ldc = args->ldc;
    float alpha = *(float*)args->alpha;
    float beta = *(float*)args->beta;
    
    // Apply beta scaling to C
    if (beta != 1.0f)
    {
        for (CBLAS_INDEX j = 0; j < n; j++)
        {
            for (CBLAS_INDEX i = 0; i < m; i++)
            {
                C(j, i) *= beta;
            }
        }
    }
    
    // Block sizes for cache optimization
    // Tuned for AVX-512 with larger registers
    CBLAS_INDEX mc = 192;  // Fits in L2 cache
    CBLAS_INDEX kc = 256;
    CBLAS_INDEX nc = 512;  // Wider for AVX-512
    
    // Allocate packed buffers
    float *packedA = (float*)malloc(mc * kc * sizeof(float));
    float *packedB = (float*)malloc(kc * nc * sizeof(float));
    
    if (!packedA || !packedB)
    {
        free(packedA);
        free(packedB);
        return;
    }
    
    // Main blocking loop
    for (CBLAS_INDEX p = 0; p < k; p += kc)
    {
        CBLAS_INDEX pb = (p + kc <= k) ? kc : (k - p);
        
        for (CBLAS_INDEX j = 0; j < n; j += nc)
        {
            CBLAS_INDEX jb = (j + nc <= n) ? nc : (n - j);
            
            for (CBLAS_INDEX i = 0; i < m; i += mc)
            {
                CBLAS_INDEX ib = (i + mc <= m) ? mc : (m - i);
                
                InnerKernel_avx512(ib, jb, pb,
                                   &A(p, i), lda,
                                   &B(j, p), ldb,
                                   &C(j, i), ldc,
                                   alpha, (i == 0),
                                   packedA, packedB);
            }
        }
    }
    
    free(packedA);
    free(packedB);
}

//------------------------------------------------------
// DGEMM AVX-512 kernel (4x16 micro-kernel for doubles)
// Each ZMM holds 8 doubles, so 16 columns = 2 ZMM
// 4 rows × 2 ZMM = 8 ZMM for C accumulator
//------------------------------------------------------
#define MR_512_D 4
#define NR_512_D 16

static void AddDot4x16_avx512_d(CBLAS_INDEX k, double *a, double *b, double *c, CBLAS_INDEX ldc, double alpha)
{
    __m512d c00, c01;
    __m512d c10, c11;
    __m512d c20, c21;
    __m512d c30, c31;
    
    __m512d b0, b1;
    __m512d a_elem;
    
    __m512d alpha_vec = _mm512_set1_pd(alpha);
    
    c00 = _mm512_setzero_pd(); c01 = _mm512_setzero_pd();
    c10 = _mm512_setzero_pd(); c11 = _mm512_setzero_pd();
    c20 = _mm512_setzero_pd(); c21 = _mm512_setzero_pd();
    c30 = _mm512_setzero_pd(); c31 = _mm512_setzero_pd();
    
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        b0 = _mm512_loadu_pd(b);
        b1 = _mm512_loadu_pd(b + 8);
        
        a_elem = _mm512_set1_pd(a[0]);
        c00 = _mm512_fmadd_pd(a_elem, b0, c00);
        c01 = _mm512_fmadd_pd(a_elem, b1, c01);
        
        a_elem = _mm512_set1_pd(a[1]);
        c10 = _mm512_fmadd_pd(a_elem, b0, c10);
        c11 = _mm512_fmadd_pd(a_elem, b1, c11);
        
        a_elem = _mm512_set1_pd(a[2]);
        c20 = _mm512_fmadd_pd(a_elem, b0, c20);
        c21 = _mm512_fmadd_pd(a_elem, b1, c21);
        
        a_elem = _mm512_set1_pd(a[3]);
        c30 = _mm512_fmadd_pd(a_elem, b0, c30);
        c31 = _mm512_fmadd_pd(a_elem, b1, c31);
        
        a += MR_512_D;
        b += NR_512_D;
    }
    
    // Apply alpha and store
    __m512d c_load;
    
    c_load = _mm512_loadu_pd(c);
    c_load = _mm512_fmadd_pd(alpha_vec, c00, c_load);
    _mm512_storeu_pd(c, c_load);
    c_load = _mm512_loadu_pd(c + 8);
    c_load = _mm512_fmadd_pd(alpha_vec, c01, c_load);
    _mm512_storeu_pd(c + 8, c_load);
    
    c_load = _mm512_loadu_pd(c + ldc);
    c_load = _mm512_fmadd_pd(alpha_vec, c10, c_load);
    _mm512_storeu_pd(c + ldc, c_load);
    c_load = _mm512_loadu_pd(c + ldc + 8);
    c_load = _mm512_fmadd_pd(alpha_vec, c11, c_load);
    _mm512_storeu_pd(c + ldc + 8, c_load);
    
    c_load = _mm512_loadu_pd(c + 2*ldc);
    c_load = _mm512_fmadd_pd(alpha_vec, c20, c_load);
    _mm512_storeu_pd(c + 2*ldc, c_load);
    c_load = _mm512_loadu_pd(c + 2*ldc + 8);
    c_load = _mm512_fmadd_pd(alpha_vec, c21, c_load);
    _mm512_storeu_pd(c + 2*ldc + 8, c_load);
    
    c_load = _mm512_loadu_pd(c + 3*ldc);
    c_load = _mm512_fmadd_pd(alpha_vec, c30, c_load);
    _mm512_storeu_pd(c + 3*ldc, c_load);
    c_load = _mm512_loadu_pd(c + 3*ldc + 8);
    c_load = _mm512_fmadd_pd(alpha_vec, c31, c_load);
    _mm512_storeu_pd(c + 3*ldc + 8, c_load);
}

static void PackMatrixA_avx512_d(CBLAS_INDEX k, double *a, CBLAS_INDEX lda, double *packedA)
{
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        double *a_ptr = &A(j, 0);
        *packedA++ = *a_ptr;
        *packedA++ = *(a_ptr + lda);
        *packedA++ = *(a_ptr + 2*lda);
        *packedA++ = *(a_ptr + 3*lda);
    }
}

static void PackMatrixB_avx512_d(CBLAS_INDEX k, double *b, CBLAS_INDEX ldb, double *packedB)
{
    for (CBLAS_INDEX i = 0; i < k; i++)
    {
        double *b_ptr = &B(0, i);
        __m512d b0 = _mm512_loadu_pd(b_ptr);
        __m512d b1 = _mm512_loadu_pd(b_ptr + 8);
        _mm512_storeu_pd(packedB, b0);
        _mm512_storeu_pd(packedB + 8, b1);
        packedB += NR_512_D;
    }
}

static void AddDot_avx512_d(CBLAS_INDEX k, double *x, CBLAS_INDEX incx, double *y, CBLAS_INDEX incy, double *gamma, double alpha)
{
    double sum = 0.0;
    for (CBLAS_INDEX p = 0; p < k; p++)
    {
        sum += x[p * incx] * y[p * incy];
    }
    *gamma += alpha * sum;
}

static void InnerKernel_avx512_d(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k,
                                  double *a, CBLAS_INDEX lda,
                                  double *b, CBLAS_INDEX ldb,
                                  double *c, CBLAS_INDEX ldc,
                                  double alpha, int first_time,
                                  double *packedA, double *packedB)
{
    CBLAS_INDEX i, j;
    
    for (j = 0; j + NR_512_D <= n; j += NR_512_D)
    {
        if (first_time)
            PackMatrixB_avx512_d(k, &B(j, 0), ldb, &packedB[j * k]);
    }
    
    for (i = 0; i + MR_512_D <= m; i += MR_512_D)
    {
        PackMatrixA_avx512_d(k, &A(0, i), lda, packedA);
        
        for (j = 0; j + NR_512_D <= n; j += NR_512_D)
        {
            AddDot4x16_avx512_d(k, packedA, &packedB[j * k], &C(j, i), ldc, alpha);
        }
        
        for (; j < n; j++)
        {
            for (CBLAS_INDEX ii = 0; ii < MR_512_D; ii++)
            {
                AddDot_avx512_d(k, packedA + ii, MR_512_D, &B(j, 0), ldb, &C(j, i + ii), alpha);
            }
        }
    }
    
    for (; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            AddDot_avx512_d(k, &A(0, i), 1, &B(j, 0), ldb, &C(j, i), alpha);
        }
    }
}

void dgemm_k_avx512(cblas_args_t* args)
{
    double *a = (double*)args->a;
    double *b = (double*)args->b;
    double *c = (double*)args->c;
    CBLAS_INDEX m = args->m;
    CBLAS_INDEX n = args->n;
    CBLAS_INDEX k = args->k;
    CBLAS_INDEX lda = args->lda;
    CBLAS_INDEX ldb = args->ldb;
    CBLAS_INDEX ldc = args->ldc;
    double alpha = *(double*)args->alpha;
    double beta = *(double*)args->beta;
    
    if (beta != 1.0)
    {
        for (CBLAS_INDEX j = 0; j < n; j++)
        {
            for (CBLAS_INDEX i = 0; i < m; i++)
            {
                C(j, i) *= beta;
            }
        }
    }
    
    CBLAS_INDEX mc = 128;
    CBLAS_INDEX kc = 256;
    CBLAS_INDEX nc = 256;
    
    double *packedA = (double*)malloc(mc * kc * sizeof(double));
    double *packedB = (double*)malloc(kc * nc * sizeof(double));
    
    if (!packedA || !packedB)
    {
        free(packedA);
        free(packedB);
        return;
    }
    
    for (CBLAS_INDEX p = 0; p < k; p += kc)
    {
        CBLAS_INDEX pb = (p + kc <= k) ? kc : (k - p);
        
        for (CBLAS_INDEX j = 0; j < n; j += nc)
        {
            CBLAS_INDEX jb = (j + nc <= n) ? nc : (n - j);
            
            for (CBLAS_INDEX i = 0; i < m; i += mc)
            {
                CBLAS_INDEX ib = (i + mc <= m) ? mc : (m - i);
                
                InnerKernel_avx512_d(ib, jb, pb,
                                      &A(p, i), lda,
                                      &B(j, p), ldb,
                                      &C(j, i), ldc,
                                      alpha, (i == 0),
                                      packedA, packedB);
            }
        }
    }
    
    free(packedA);
    free(packedB);
}

#else

// Stub implementations for non-AVX512 builds
void sgemm_k_avx512(cblas_args_t* args) { (void)args; }
void dgemm_k_avx512(cblas_args_t* args) { (void)args; }

#endif
