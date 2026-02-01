//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

//======================================================================
// CACHE-BLOCKING STRATEGY FOR GEMM (General Matrix Multiply)
//======================================================================
//
// This implementation uses a three-level loop tiling strategy inspired by
// the GotoBLAS algorithm (see "Anatomy of High-Performance Matrix Multiplication"
// by Goto and van de Geijn, ACM TOMS 2008). The goal is to maximize cache reuse
// by carefully orchestrating data movement through the memory hierarchy.
//
// MEMORY HIERARCHY OVERVIEW:
// ---------------------------
// Modern CPUs have a cache hierarchy with different capacities and speeds:
//   L1 data cache: 32-64KB (very fast, ~4 cycles)
//   L2 cache:     256KB-1MB (fast, ~12 cycles)  
//   L3 cache:     2-32MB (moderate, ~40 cycles)
//   Main memory:  GB+ (slow, ~100+ cycles)
//
// CACHE-BLOCKING ALGORITHM:
// -------------------------
// The algorithm computes C = A × B by blocking the matrices into tiles that
// fit in cache. The computation proceeds in three nested loops:
//
//   for each kc-sized panel of k (outer k-loop)
//     for each mc-sized panel of m (outer m-loop)
//       Pack a mc×kc tile of A into contiguous buffer (packedA)
//       for each nb-sized panel of n (inner n-loop, inside InnerKernel)
//         Pack a kc×nb tile of B into contiguous buffer (packedB)
//         Compute the mc×nb tile of C using packed data
//
// TILE SIZE PARAMETERS:
// ---------------------
// The three tile dimensions (mc, kc, nb) are runtime-determined based on
// detected cache sizes (see cblas_compute_gemm_block_sizes() in util.c):
//
//   mc: Number of rows of A to pack (typically 128-512)
//       - Controls the outer m-loop blocking
//       - Packed A buffer size: mc × kc × 4 bytes
//
//   kc: Inner dimension (typically 128-256)
//       - Controls the outer k-loop blocking
//       - Shared dimension for both packed buffers
//       - Critical for cache hit rate
//
//   nb: Number of columns of B to pack (typically 256-1024)
//       - Controls how much of C is computed at once
//       - Packed B buffer size: kc × nb × 4 bytes
//
// WHY THESE SIZES?
// ----------------
// The goal is to keep both packedA and packedB resident in L2 cache:
//
//   Total packed data = (mc × kc + kc × nb) × 4 bytes
//
// For typical defaults (mc=256, kc=256, nb=512):
//   Packed A: 256 × 256 × 4 = 256 KB
//   Packed B: 256 × 512 × 4 = 512 KB
//   Total:                    768 KB  (~85% of a 1MB L2 cache)
//
// The 85% target provides headroom for other data (output matrix C, stack, etc.)
// while maximizing cache utilization.
//
// RELATIONSHIP TO CACHE LEVELS:
// ------------------------------
// L1 Cache: The 4×4 micro-kernel (AddDot4x4) is designed to keep 4 rows of A
//           and 4 columns of B in L1 cache or registers during computation.
//
// L2 Cache: Packed A and B tiles are sized to fit in L2, enabling high-bandwidth
//           access during the micro-kernel execution. This is the critical
//           optimization point.
//
// L3 Cache: The original matrices A, B, C may partially fit in L3, but the
//           algorithm doesn't rely on this - it's designed to work even when
//           matrices exceed all cache levels.
//
// TUNING FOR DIFFERENT ARCHITECTURES:
// ------------------------------------
// Tile sizes are auto-tuned based on detected L1 and L2 cache sizes:
//
// 1. Apple M-series (128KB L1d, 8-24MB L2):
//    mc=512, kc=256, nb=512 (larger tiles for bigger caches)
//
// 2. ARM Cortex (64KB L1d, 512KB-2MB L2):
//    mc=256, kc=256, nb=512 (balanced tiles)
//
// 3. Intel/AMD (32KB L1d, 256KB-1MB L2):
//    mc=192, kc=256, nb=384 (fits in smaller L2)
//
// 4. Older/embedded CPUs (16-32KB L1d, 128-256KB L2):
//    mc=128, kc=192, nb=256 (conservative sizes)
//
// The auto-tuning code in util.c:cblas_compute_gemm_block_sizes() implements
// this logic and enforces that total packed data ≤ 85% of L2 cache.
//
// MANUAL TUNING:
// --------------
// To manually tune for your architecture:
// 1. Measure L2 cache size: `cat /proc/cpuinfo` (Linux) or use utilities
// 2. Set tile sizes such that (mc×kc + kc×nb)×4 ≤ 0.85 × L2_size_bytes
// 3. Prefer larger mc and nb for better parallelism (divisible by 16)
// 4. kc affects both buffers, so it's usually kept moderate (128-256)
// 5. Run gemm_perf to measure GFlops and iterate
//
// TRADEOFFS:
// ----------
// Larger tiles:
//   + Better amortization of packing overhead
//   + More work per cache miss
//   + Better vectorization opportunities
//   - Higher risk of cache eviction
//   - More memory required for packed buffers
//   - May not fit in smaller caches
//
// Smaller tiles:
//   + Guaranteed to fit in cache
//   + Lower memory footprint
//   + Works on more architectures
//   - More packing overhead
//   - Less opportunity for SIMD/parallelism
//   - More loop iterations
//
// REFERENCES:
// -----------
// [1] Goto, K. and van de Geijn, R. (2008). "Anatomy of High-Performance 
//     Matrix Multiplication". ACM Transactions on Mathematical Software, 34(3).
//     https://www.cs.utexas.edu/~flame/pubs/GotoTOMS2008.pdf
//
// [2] Low, T.M., et al. (2016). "Analytical Modeling Is Enough for High-Performance
//     BLIS". ACM TOMS, 43(2).
//
// [3] How to Optimize GEMM (GotoBLAS/BLIS tutorial):
//     https://github.com/flame/how-to-optimize-gemm/wiki
//
//======================================================================

// Matrix sub-tile block sizes for caching data in contiguous memory
// These are now runtime-determined based on cache size (see cblas_gemm_mc, cblas_gemm_kc, cblas_gemm_nb)
// Maximum buffer sizes for static allocation (when USE_STATIC_BUFFERS is defined)
#define MAX_MC 512
#define MAX_KC 256
#define MAX_NB 1024

// Prefetch distance tuning - prefetch this many iterations ahead
#define PREFETCH_DISTANCE 16

// macros to simpify matrix element access
#define A(col, row) a[((row) * lda + (col))]
#define B(col, row) b[((row) * ldb + (col))]
#define C(col, row) c[((row) * ldc + (col))]

//#define X(i) x[(i) * incx]
#define Y(i) y[(i) * incx]

#define CHECK_GUARDS()  assert((unsigned)aguard == 0xbaadf00d && (unsigned)bguard == 0xbaadf00d && (unsigned)cguard == 0xbaadf00d)

#ifdef USE_STATIC_BUFFERS
    static int aguard CBLAS_UNUSED = 0xbaadf00d;
    static float packedA[MAX_MC * MAX_KC];
    static int bguard CBLAS_UNUSED = 0xbaadf00d;
    static float packedB[MAX_KC * MAX_NB];
    static int cguard CBLAS_UNUSED = 0xbaadf00d;
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
// compute 16 dot products at a time, 4 cols x 4 rows (non-FMA version)
//------------------------------------------------------
static void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    __m128 c_row1, c_row2, c_row3, c_row4;
    __m128 b_row;
    __m128 a_p0, a_p1, a_p2, a_p3;
    
    // Use unaligned loads for c matrix since it may not be 16-byte aligned
    c_row1 = _mm_loadu_ps(&C(0,0));
    c_row2 = _mm_loadu_ps(&C(0,1));
    c_row3 = _mm_loadu_ps(&C(0,2));
    c_row4 = _mm_loadu_ps(&C(0,3));

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // load and duplicate 
        a_p0 = _mm_load_ps1(a);
        a_p1 = _mm_load_ps1(a + 1);
        a_p2 = _mm_load_ps1(a + 2);
        a_p3 = _mm_load_ps1(a + 3);

        // Prefetch data ahead (after current load, before pointer update)
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * 4), 0, 3);
        }

        a += 4;

        // Use unaligned load for b since alignment is not guaranteed
        b_row = _mm_loadu_ps(b);

        b += 4;

        // rows 1 - 4 using SSE3
        c_row1 = _mm_add_ps(c_row1, _mm_mul_ps(a_p0, b_row));
        c_row2 = _mm_add_ps(c_row2, _mm_mul_ps(a_p1, b_row));
        c_row3 = _mm_add_ps(c_row3, _mm_mul_ps(a_p2, b_row));
        c_row4 = _mm_add_ps(c_row4, _mm_mul_ps(a_p3, b_row));
    }

    // Use unaligned stores for c matrix since it may not be 16-byte aligned
    _mm_storeu_ps(&C(0, 0), c_row1);
    _mm_storeu_ps(&C(0, 1), c_row2);
    _mm_storeu_ps(&C(0, 2), c_row3);
    _mm_storeu_ps(&C(0, 3), c_row4);
}

//------------------------------------------------------
// compute 16 dot products at a time, 4 cols x 4 rows (FMA version)
//------------------------------------------------------
static void AddDot4x4_fma(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    __m128 c_row1, c_row2, c_row3, c_row4;
    __m128 b_row;
    __m128 a_p0, a_p1, a_p2, a_p3;
    
    // Use unaligned loads for c matrix since it may not be 16-byte aligned
    c_row1 = _mm_loadu_ps(&C(0,0));
    c_row2 = _mm_loadu_ps(&C(0,1));
    c_row3 = _mm_loadu_ps(&C(0,2));
    c_row4 = _mm_loadu_ps(&C(0,3));

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // load and duplicate 
        a_p0 = _mm_load_ps1(a);
        a_p1 = _mm_load_ps1(a + 1);
        a_p2 = _mm_load_ps1(a + 2);
        a_p3 = _mm_load_ps1(a + 3);

        // Prefetch data ahead (after current load, before pointer update)
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

    // Use unaligned stores for c matrix since it may not be 16-byte aligned
    _mm_storeu_ps(&C(0, 0), c_row1);
    _mm_storeu_ps(&C(0, 1), c_row2);
    _mm_storeu_ps(&C(0, 2), c_row3);
    _mm_storeu_ps(&C(0, 3), c_row4);
}

#elif defined(__aarch64__) && defined(USE_SIMD)

static void AddDot4x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    float32x4_t c_row1, c_row2, c_row3, c_row4;
    float32x4_t b_row;
    float32x4_t a_p0, a_p1, a_p2, a_p3;
    
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

        // Prefetch data ahead (after current load, before pointer update)
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(b + (PREFETCH_DISTANCE * 4), 0, 3);
        }

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

        // Prefetch data ahead (after current load, before pointer update)
        if (p + PREFETCH_DISTANCE < k) {
            CBLAS_PREFETCH(a + (PREFETCH_DISTANCE * 4), 0, 3);
            CBLAS_PREFETCH(&B(0, p + PREFETCH_DISTANCE), 0, 3);
            CBLAS_PREFETCH(&B(1, p + PREFETCH_DISTANCE), 0, 3);
            CBLAS_PREFETCH(&B(2, p + PREFETCH_DISTANCE), 0, 3);
            CBLAS_PREFETCH(&B(3, p + PREFETCH_DISTANCE), 0, 3);
        }

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
CBLAS_UNUSED static void AddDot1x4(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *b, CBLAS_INDEX ldb, float *c, CBLAS_INDEX ldc)
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
// PackMatrixB - Copy a k×4 panel of B into contiguous memory
//------------------------------------------------------
// PURPOSE:
//   Transform B data from strided (ldb) layout to contiguous layout for
//   better cache performance during the micro-kernel execution.
//
// WHY PACKING?
//   - Original B has stride ldb, causing cache misses when accessed
//   - Packed B is contiguous, enabling sequential access and prefetching
//   - Packing cost is amortized over many reuses in the micro-kernel
//   - Packed data stays in L2 cache for fast repeated access
//
// PARAMETERS:
//   k: Number of rows to pack (≤ kc)
//   b: Pointer to source data in B matrix (with stride ldb)
//   ldb: Leading dimension of B (stride between rows)
//   b_to: Pointer to packed buffer (contiguous storage)
//
// LAYOUT:
//   Packs 4 consecutive columns of B (each with k rows) into sequential memory.
//   This matches the 4-column micro-panel size used by AddDot4x4.
//
static void PackMatrixB(CBLAS_INDEX k, float *b, CBLAS_INDEX ldb, float *b_to)
{
    // loop over rows of B with prefetching
    for (CBLAS_INDEX j = 0; j < k; j++)
    {
        float *b_ij_pntr = &B(0, j);

        // Prefetch ahead for next iterations to hide memory latency
        if (j + 8 < k) {
            CBLAS_PREFETCH(&B(0, j + 8), 0, 3);
        }

        // Copy 4 consecutive elements (one row, 4 columns)
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
// PURPOSE:
//   Transform A data from strided (lda) layout to contiguous layout for
//   better cache performance during the micro-kernel execution.
//
// WHY PACKING?
//   - Original A has stride lda between rows, causing non-sequential access
//   - Packed A is contiguous in memory, improving cache line utilization
//   - Packing cost is amortized over many columns of C computed
//   - Packed data stays in L2 cache for fast repeated access
//
// PARAMETERS:
//   k: Number of columns to pack (≤ kc)
//   a: Pointer to source data in A matrix (with stride lda)
//   lda: Leading dimension of A (stride between rows)
//   a_to: Pointer to packed buffer (contiguous storage)
//
// LAYOUT:
//   Packs 4 consecutive rows of A (each with k columns) into sequential memory.
//   This matches the 4-row micro-panel size used by AddDot4x4.
//
static void PackMatrixA(CBLAS_INDEX k, float *a, CBLAS_INDEX lda, float *a_to)
{
    CBLAS_INDEX i;
    float   *a_0i_pntr = &A(0,0), *a_1i_pntr = &A(0,1),
            *a_2i_pntr = &A(0,2), *a_3i_pntr = &A(0,3);

    // loop over cols of A with prefetching
    for (i = 0; i < k; i++)
    {
        // Prefetch ahead for next iterations to hide memory latency
        if (i + 8 < k) {
            CBLAS_PREFETCH(a_0i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_1i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_2i_pntr + 8, 0, 3);
            CBLAS_PREFETCH(a_3i_pntr + 8, 0, 3);
        }

        // Copy one element from each of 4 rows (interleaved for cache efficiency)
        *a_to       = *a_0i_pntr++;
        *(a_to + 1) = *a_1i_pntr++;
        *(a_to + 2) = *a_2i_pntr++;
        *(a_to + 3) = *a_3i_pntr++;

        a_to += 4;
    }
}

//------------------------------------------------------
// GEMM InnerKernel - The core computation engine
//------------------------------------------------------
// This function implements the innermost part of the cache-blocking algorithm.
// It processes a single mc×n tile of the output matrix C.
//
// PARAMETERS:
//   m: Number of rows in this tile (≤ mc)
//   n: Number of columns in full output (all of C's width)
//   k: Inner dimension for this iteration (≤ kc)
//   a: Pointer to mc×k tile of A
//   b: Pointer to k×n tile of B
//   c: Pointer to mc×n tile of C (output)
//
// ALGORITHM:
//   1. Pack B: Copy k×nb blocks of B into contiguous buffer (packedB)
//   2. For each 4-row block of the m rows:
//      a. Pack A: Copy 4×k block of A into contiguous buffer (packedA)
//      b. For each 4-column block of the n columns:
//         - Compute 4×4 micro-tile using SIMD (AddDot4x4)
//         - This is where the actual FLOPs happen!
//      c. Handle leftover columns (< 4) with scalar code
//   3. Handle leftover rows (< 4) with scalar code
//
// CACHE BEHAVIOR:
//   - packedA and packedB fit in L2 cache by design
//   - The 4×4 micro-kernel keeps working set in L1 cache/registers
//   - Data is accessed sequentially from packed buffers (good prefetching)
//   - C is accessed with stride ldc (may cause cache misses, but write-back)
//
static void InnerKernel(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc)
{
#if !defined(USE_STATIC_BUFFERS)
    #ifdef _WIN32
        int aguard = 0xbaadf00d;
        float* packedA = _malloca(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = _malloca(cblas_gemm_kc * cblas_gemm_nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #else
        int aguard = 0xbaadf00d;
        float* packedA = alloca(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = alloca(cblas_gemm_kc * cblas_gemm_nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #endif
#endif
    CHECK_GUARDS();

//printf("tile: (%ld, %ld) x (%ld, %ld)\n", k, m, n, k);

    //int row_leftover    = m % 4; 
    //int col_leftover    = n % 4;
    CBLAS_INDEX row, col;

    // INNER N-LOOP: Process rows of C in 4-row blocks (micro-panel height)
    // This loop unrolls the row dimension to enable 4×4 SIMD micro-kernels
    for (row = 0; row + 4 <= m; row += 4)
    {
        // Pack matrix B once per row-block (amortizes packing cost)
        // packedB contains k×n data in column-major order for cache-friendly access
        if (row == 0)
            PackMatrixB(k, &B(0, 0), ldb, packedB);

        // Process columns of C in 4-column blocks (micro-panel width)
        for (col = 0; col + 4 <= n; col += 4)
        {
            // Pack matrix A once per column-block (amortizes packing cost)
            // packedA contains 4×k data in row-major order for cache-friendly access
            if (col == 0) 
                PackMatrixA(k, &A(0, row), lda, packedA);

            // Compute 4×4 tile of C using highly optimized SIMD micro-kernel
            // This is the performance-critical inner loop (>90% of compute time)
            AddDot4x4(k, packedA, 4, packedB, k, &C(col, row), ldc);
        }

        // handle leftover columns (when n is not divisible by 4)
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
                //AddDot1x4(k, &A(0, row), lda, &B(col, 0), ldb, &C(col, row), ldc);
                AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
                AddDot(k, &A(0, row+1), 1, &B(col, 0), ldb, &C(col, row+1));
                AddDot(k, &A(0, row+2), 1, &B(col, 0), ldb, &C(col, row+2));
                AddDot(k, &A(0, row+3), 1, &B(col, 0), ldb, &C(col, row+3));
                CBLAS_FALLTHROUGH;
            case 0: ;   // nothing to do!
        }
    }

    // handle leftover rows (when m is not divisible by 4)    
    switch(m - row)
    {
        case 3:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 2), 1, &B(col, 0), ldb, &C(col, row + 2));
            CBLAS_FALLTHROUGH;
        case 2:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1));
            CBLAS_FALLTHROUGH;
        case 1:    for (col = 0; col < n; col++) AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
            CBLAS_FALLTHROUGH;
        case 0: ;   // nothing to do!
    }

    CHECK_GUARDS();
}

//------------------------------------------------------
// GEMM kernel (non-FMA)
//------------------------------------------------------
void sgemm_k(cblas_args_t* args)
{
    InnerKernel(args->ib, args->n, args->pb, args->a, args->lda, args->b, args->ldb, args->c, args->ldc);
}

#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

//------------------------------------------------------
// GEMM kernel with FMA support (x86-64)
//------------------------------------------------------
static void InnerKernel_fma(CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float* c, CBLAS_INDEX ldc)
{
#if !defined(USE_STATIC_BUFFERS)
    #ifdef _WIN32
        int aguard = 0xbaadf00d;
        float* packedA = _malloca(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = _malloca(cblas_gemm_kc * cblas_gemm_nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #else
        int aguard = 0xbaadf00d;
        float* packedA = alloca(cblas_gemm_mc * cblas_gemm_kc * sizeof(float));
        int bguard = 0xbaadf00d;
        float* packedB = alloca(cblas_gemm_kc * cblas_gemm_nb * sizeof(float));
        int cguard = 0xbaadf00d;
    #endif
#endif
    CHECK_GUARDS();

    CBLAS_INDEX row, col;

    // Loop over the rows and columns of C unrolled by 4
    for (row = 0; row + 4 <= m; row += 4)
    {
        // we are pre-caching all cols so we need to do it only once
        if (row == 0)
            PackMatrixB(k, &B(0, 0), ldb, packedB);

        for (col = 0; col + 4 <= n; col += 4)
        {
            // we are pre-caching all rows so we need to do it only once
            if (col == 0) 
                PackMatrixA(k, &A(0, row), lda, packedA);

            AddDot4x4_fma(k, packedA, 4, packedB, k, &C(col, row), ldc);
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
            case 0: ;   // nothing to do!
        }
    }

    // handle leftover rows    
    switch(m - row)
    {
        case 3:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 2), 1, &B(col, 0), ldb, &C(col, row + 2));
            CBLAS_FALLTHROUGH;
        case 2:    for (col = 0; col < n; col++) AddDot(k, &A(0, row + 1), 1, &B(col, 0), ldb, &C(col, row + 1));
            CBLAS_FALLTHROUGH;
        case 1:    for (col = 0; col < n; col++) AddDot(k, &A(0, row), 1, &B(col, 0), ldb, &C(col, row));
            CBLAS_FALLTHROUGH;
        case 0: ;   // nothing to do!
    }

    CHECK_GUARDS();
}

//------------------------------------------------------
// GEMM kernel wrapper (FMA version)
//------------------------------------------------------
void sgemm_k_fma(cblas_args_t* args)
{
    InnerKernel_fma(args->ib, args->n, args->pb, args->a, args->lda, args->b, args->ldb, args->c, args->ldc);
}

#endif

//------------------------------------------------------
// single-precision general matrix multiply
//------------------------------------------------------
void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
    (void)alpha;
    (void)beta;
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
    if (!a)
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

    CBLAS_STATS_START();

    CBLAS_INDEX pb, ib;

#if defined(MT_ENABLED)
    int mt_used = (m * n * k > CBLAS_MT_GEMM) ? 1 : 0;
    
    if (mt_used)
    {
        CBLAS_INDEX horiz_tiles = k / cblas_gemm_kc + 1;
        CBLAS_INDEX vert_tiles = m / cblas_gemm_mc + 1;
        CBLAS_INDEX total_tiles = horiz_tiles * vert_tiles;
        CBLAS_INDEX tile_count = 0;

        //printf("tile count = %u\n", total_tiles);
        // Use heap allocation for large tile arrays to avoid stack overflow
        work_queue_t *queue = (work_queue_t*)malloc(total_tiles * sizeof(work_queue_t));
        cblas_args_t *args = (cblas_args_t*)malloc(total_tiles * sizeof(cblas_args_t));
        
        if (!queue || !args) {
            free(queue);
            free(args);
            return;  // Out of memory
        }

        // THREE-LEVEL CACHE-BLOCKING LOOP STRUCTURE (Multi-threaded):
        // ============================================================
        //
        // The same three-level loop structure as single-threaded version, but
        // each tile is enqueued as an independent task for parallel execution.
        // This allows multiple threads to work on different mc×n tiles simultaneously.
        //
        // Outer k-loop: Iterate over k dimension in kc-sized blocks
        // Middle m-loop: Iterate over m dimension in mc-sized blocks
        // - Each combination creates a work item for the thread pool
        for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
        {
            pb = MIN(k - p, cblas_gemm_kc);  // Handle edge case when k % kc != 0
            for (CBLAS_INDEX row = 0; row < m; row += cblas_gemm_mc) 
            {
                ib = MIN(m - row, cblas_gemm_mc);  // Handle edge case when m % mc != 0

                // Setup work item for this mc×n tile
                // Each thread will independently pack and compute its assigned tile
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

        // synchronously execute task queue (parallel execution by worker threads)
        cblas_execute(tile_count, queue);
        
        // Free heap-allocated arrays
        free(queue);
        free(args);
    }
    else
    {
        // Below threshold, use single-threaded implementation
        // Runtime dispatch: check for FMA support once and cache result
        static int fma_available = -1;
        if (fma_available == -1) {
            unsigned int features = cpu_get_features();
            fma_available = (features & CPU_x64_FMA3) ? 1 : 0;
        }
        
        // THREE-LEVEL CACHE-BLOCKING LOOP STRUCTURE:
        // ===========================================
        //
        // Outer k-loop: Iterate over k dimension in kc-sized blocks
        // - Each iteration processes a kc-wide vertical panel of A and horizontal panel of B
        // - This loop determines how many times we need to pack matrix data
        for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
        {
            pb = MIN(k - p, cblas_gemm_kc);  // Handle edge case when k % kc != 0
            
            // Middle m-loop: Iterate over m dimension in mc-sized blocks
            // - Each iteration processes an mc-tall horizontal panel of A
            // - packedA is reused for all columns of C in the inner kernel
            for (CBLAS_INDEX row = 0; row < m; row += cblas_gemm_mc) 
            {
                ib = MIN(m - row, cblas_gemm_mc);  // Handle edge case when m % mc != 0
                
                // InnerKernel handles the n-loop (nb-sized blocks) and performs:
                // 1. Pack mc×kc tile of A into contiguous buffer (once per m-block)
                // 2. Pack kc×nb tile of B into contiguous buffer (once per n-block)
                // 3. Compute mc×nb tile of C using packed data (micro-kernel)
                //
                // The packed data stays in L2 cache, enabling high-bandwidth access
                // during the intensive computation phase.
#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
                if (fma_available) {
                    InnerKernel_fma(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
                } else
#endif
                {
                    InnerKernel(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
                }
            }
        }
    }

#else
    int mt_used = 0;
    // Compute an mc x n block of C by a call to the InnerKernel
    // Runtime dispatch: check for FMA support once and cache result
    static int fma_available = -1;
    if (fma_available == -1) {
        unsigned int features = cpu_get_features();
        fma_available = (features & CPU_x64_FMA3) ? 1 : 0;
    }
    
    for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
    {
        pb = MIN(k - p, cblas_gemm_kc);
        for (CBLAS_INDEX row = 0; row < m; row += cblas_gemm_mc) 
        {
            ib = MIN(m - row, cblas_gemm_mc);
#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
            if (fma_available) {
                InnerKernel_fma(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
            } else
#endif
            {
                InnerKernel(ib, n, pb, &A(p, row), lda, &B(0, p), ldb, &C(0, row), ldc);
            }
        }
    }
#endif

    CBLAS_STATS_END("sgemm", m * n * k, mt_used);
}

//------------------------------------------------------
// single-precision reference matrix multipl
//------------------------------------------------------
void cblas_sgemm_naive(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, float alpha, float* a, CBLAS_INDEX lda, float* b, CBLAS_INDEX ldb, float beta, float* c, CBLAS_INDEX ldc)
{
    (void)layout;  // Used only in assert
    (void)transa;  // Used only in assert
    (void)transb;  // Used only in assert
    (void)alpha;   // Used only in assert
    (void)beta;    // Used only in assert
    assert(layout == CblasRowMajor || layout == CblasColMajor);
    assert(transa == CblasTrans || transa == CblasNoTrans);
    assert(transb == CblasTrans || transb == CblasNoTrans);
    assert(m > 0 && n > 0 && k > 0);
    assert(a && b && c);
    assert(alpha != 0.0f && beta != 0.0f);

    for (CBLAS_INDEX row = 0; row < m; row++)
        for (CBLAS_INDEX col = 0; col < n; col++)
            for (CBLAS_INDEX p = 0; p < k; p++)
                C(col, row) += A(p, row) * B(col, p);
}

//------------------------------------------------------
// Double-precision SIMD: compute 4 dot products at a time, 2 cols x 2 rows
// NOTE: Currently unused. Reserved for future packed-matrix SIMD implementation
// similar to sgemm's AddDot4x4. Requires matrix packing before use.
//------------------------------------------------------
#if defined(USE_SSE) && defined(USE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))

CBLAS_UNUSED static void AddDot2x2_d(CBLAS_INDEX k, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    __m128d c_row1, c_row2;
    __m128d b_row;
    __m128d a_p0, a_p1;
    
    // Load initial C values
    c_row1 = _mm_loadu_pd(&C(0,0));
    c_row2 = _mm_loadu_pd(&C(0,1));

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // Load and duplicate 
        a_p0 = _mm_load_pd1(a);
        a_p1 = _mm_load_pd1(a + 1);

        a += 2;

        // Load B row
        b_row = _mm_loadu_pd(b);
        b += 2;

#if defined(USE_INTEL_FMA)
        // Rows 1-2 using FMA
        c_row1 = _mm_fmadd_pd(a_p0, b_row, c_row1);
        c_row2 = _mm_fmadd_pd(a_p1, b_row, c_row2);
#else
        // Rows 1-2 using SSE
        c_row1 = _mm_add_pd(c_row1, _mm_mul_pd(a_p0, b_row));
        c_row2 = _mm_add_pd(c_row2, _mm_mul_pd(a_p1, b_row));
#endif
    }

    // Store results
    _mm_storeu_pd(&C(0, 0), c_row1);
    _mm_storeu_pd(&C(0, 1), c_row2);
}

#elif defined(__aarch64__) && defined(__ARM_NEON) && defined(USE_SIMD)

CBLAS_UNUSED static void AddDot2x2_d(CBLAS_INDEX k, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double *c, CBLAS_INDEX ldc)
{
    (void)lda;
    (void)ldb;
    float64x2_t c_row1, c_row2;
    float64x2_t b_row;
    float64x2_t a_p0, a_p1;
    
    // Load initial C values
    c_row1 = vld1q_f64(&C(0,0));
    c_row2 = vld1q_f64(&C(0,1));

	for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        // Load and duplicate 
        a_p0 = vld1q_dup_f64(a);
        a_p1 = vld1q_dup_f64(a + 1);

        a += 2;

        // Load B row
        b_row = vld1q_f64(b);
        b += 2;

#ifdef __ARM_FEATURE_FMA
        // Rows 1-2 using FMA
        c_row1 = vfmaq_f64(c_row1, a_p0, b_row);
        c_row2 = vfmaq_f64(c_row2, a_p1, b_row);
#else
        // Rows 1-2 using NEON MUL and ADD
        c_row1 = vaddq_f64(c_row1, vmulq_f64(a_p0, b_row));
        c_row2 = vaddq_f64(c_row2, vmulq_f64(a_p1, b_row));
#endif
    }

    // Store results
    vst1q_f64(&C(0, 0), c_row1);
    vst1q_f64(&C(0, 1), c_row2);
}

#else   // fall-back non-SIMD version

CBLAS_UNUSED static void AddDot2x2_d(CBLAS_INDEX k, double* a, CBLAS_INDEX lda, double* b, CBLAS_INDEX ldb, double* c, CBLAS_INDEX ldc)
{
    register double c_00, c_10, c_01, c_11;
    register double a_p0, a_p1;
    register double b_0p, b_1p;

    c_00 = C(0, 0); c_10 = C(1, 0);
    c_01 = C(0, 1); c_11 = C(1, 1);

    for (CBLAS_INDEX p = 0; p < k; p++) 
    {
        a_p0 = A(p, 0);
        a_p1 = A(p, 1);

        b_0p = B(0, p);
        b_1p = B(1, p);

        c_00 += a_p0 * b_0p;
        c_10 += a_p0 * b_1p;

        c_01 += a_p1 * b_0p;
        c_11 += a_p1 * b_1p;
    }

    C(0, 0) = c_00; C(1, 0) = c_10;
    C(0, 1) = c_01; C(1, 1) = c_11;
}

#endif

//------------------------------------------------------
// Double-precision scalar dot product (1x1 block)
//------------------------------------------------------
CBLAS_UNUSED static void AddDot_d(CBLAS_INDEX k, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double *gamma)
{
    (void)lda;
    (void)ldb;
    
	for (CBLAS_INDEX p = 0; p < k; p++)
    {
		*gamma += (*a) * (*b);
        a++;
        b++;
	}
}

//------------------------------------------------------
// double-precision general matrix multiply
//------------------------------------------------------
void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, CBLAS_INDEX m, CBLAS_INDEX n, CBLAS_INDEX k, double alpha, double *a, CBLAS_INDEX lda, double *b, CBLAS_INDEX ldb, double beta, double *c, CBLAS_INDEX ldc)
{
    (void)alpha;
    (void)beta;
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
    if (!a)
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
    if (!a || !b || !c || lda < MAX(1, nrowa) || ldb < MAX(1, nrowb) || ldc < MAX(1, m))
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

    CBLAS_STATS_START();

    int mt_used = 0;

    // Apply beta scaling to C if needed
    if (beta == 0.0)
    {
        // Zero out C
        for (CBLAS_INDEX row = 0; row < m; row++)
            for (CBLAS_INDEX col = 0; col < n; col++)
                C(col, row) = 0.0;
    }
    else if (beta != 1.0)
    {
        // Scale C by beta
        for (CBLAS_INDEX row = 0; row < m; row++)
            for (CBLAS_INDEX col = 0; col < n; col++)
                C(col, row) *= beta;
    }

    // If alpha is 0, we're done (C = beta * C)
    if (alpha == 0.0)
    {
        CBLAS_STATS_END("dgemm", m * n * k, mt_used);
        return;
    }

    // Simple SIMD-optimized implementation using blocked approach
    // Process in 2x2 blocks for better cache locality
    CBLAS_INDEX row, col;
    
    if (alpha == 1.0)
    {
        // Optimized path when alpha = 1.0
        for (row = 0; row + 2 <= m; row += 2)
        {
            for (col = 0; col + 2 <= n; col += 2)
            {
                // Process 2x2 block
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    // Row 0, Cols 0-1
                    C(col, row) += A(p, row) * B(col, p);
                    C(col + 1, row) += A(p, row) * B(col + 1, p);
                    
                    // Row 1, Cols 0-1
                    C(col, row + 1) += A(p, row + 1) * B(col, p);
                    C(col + 1, row + 1) += A(p, row + 1) * B(col + 1, p);
                }
            }
            
            // Handle remaining columns
            for (; col < n; col++)
            {
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    C(col, row) += A(p, row) * B(col, p);
                    C(col, row + 1) += A(p, row + 1) * B(col, p);
                }
            }
        }
        
        // Handle remaining rows
        for (; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                for (CBLAS_INDEX p = 0; p < k; p++)
                    C(col, row) += A(p, row) * B(col, p);
            }
        }
    }
    else
    {
        // Generic path with alpha scaling
        for (row = 0; row + 2 <= m; row += 2)
        {
            for (col = 0; col + 2 <= n; col += 2)
            {
                // Process 2x2 block
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    // Row 0, Cols 0-1
                    C(col, row) += alpha * A(p, row) * B(col, p);
                    C(col + 1, row) += alpha * A(p, row) * B(col + 1, p);
                    
                    // Row 1, Cols 0-1
                    C(col, row + 1) += alpha * A(p, row + 1) * B(col, p);
                    C(col + 1, row + 1) += alpha * A(p, row + 1) * B(col + 1, p);
                }
            }
            
            // Handle remaining columns
            for (; col < n; col++)
            {
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    C(col, row) += alpha * A(p, row) * B(col, p);
                    C(col, row + 1) += alpha * A(p, row + 1) * B(col, p);
                }
            }
        }
        
        // Handle remaining rows
        for (; row < m; row++)
        {
            for (col = 0; col < n; col++)
            {
                for (CBLAS_INDEX p = 0; p < k; p++)
                    C(col, row) += alpha * A(p, row) * B(col, p);
            }
        }
    }

    CBLAS_STATS_END("dgemm", m * n * k, mt_used);
}

