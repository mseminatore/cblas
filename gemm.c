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
//     https://doi.org/10.1145/2870650
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
        return;
    }
#endif
#endif

    CBLAS_STATS_START();

    // Handle beta scaling of C before computing A*B
    // C = beta * C (done once, before any kernel calls)
    if (beta == 0.0f)
    {
        // Zero out C
        for (CBLAS_INDEX row = 0; row < m; row++)
            for (CBLAS_INDEX col = 0; col < n; col++)
                C(col, row) = 0.0f;
    }
    else if (beta != 1.0f)
    {
        // Scale C by beta
        for (CBLAS_INDEX row = 0; row < m; row++)
            for (CBLAS_INDEX col = 0; col < n; col++)
                C(col, row) *= beta;
    }
    // If beta == 1.0f, C is unchanged

    // If alpha is 0, we're done (C = beta * C)
    if (alpha == 0.0f)
    {
        CBLAS_STATS_END("sgemm", m * n * k, 0);
        return;
    }

    CBLAS_INDEX pb, ib;

#if defined(MT_ENABLED)
    int mt_used = (m * n * k > CBLAS_MT_GEMM) ? 1 : 0;
    
    if (mt_used)
    {
        CBLAS_INDEX horiz_tiles = k / cblas_gemm_kc + 1;
        CBLAS_INDEX vert_tiles = m / cblas_gemm_mc + 1;
        CBLAS_INDEX total_tiles = horiz_tiles * vert_tiles;
        CBLAS_INDEX tile_count = 0;

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
                args[tile_count].alpha_s = alpha;  // Pass alpha to kernel
                args[tile_count].beta_s = 1.0f;    // Beta already applied to C
            
                queue[tile_count].finished   = 0;
                queue[tile_count].args       = &args[tile_count];
                queue[tile_count].kernel     = blas_kernels.sgemm_k;
                queue[tile_count].next       = &queue[tile_count + 1];

                tile_count++;
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
        // Below threshold, use single-threaded implementation via dispatched kernel
        cblas_args_t st_args;
        st_args.incx = 1;
        st_args.incy = 1;
        st_args.n = n;
        st_args.lda = lda;
        st_args.ldb = ldb;
        st_args.ldc = ldc;
        st_args.alpha_s = alpha;  // Pass alpha to kernel
        st_args.beta_s = 1.0f;    // Beta already applied to C
        st_args.thread_id = 0;    // Main thread uses buffer slot 0
        
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
                
                // Setup args for this tile
                st_args.a = &A(p, row);
                st_args.b = &B(0, p);
                st_args.c = &C(0, row);
                st_args.ib = ib;
                st_args.pb = pb;
                
                // Dispatch to platform-optimized kernel
                blas_kernels.sgemm_k(&st_args);
            }
        }
    }

#else
    int mt_used = 0;
    // Single-threaded implementation via dispatched kernel
    cblas_args_t st_args;
    st_args.incx = 1;
    st_args.incy = 1;
    st_args.n = n;
    st_args.lda = lda;
    st_args.ldb = ldb;
    st_args.ldc = ldc;
    st_args.alpha_s = alpha;  // Pass alpha to kernel
    st_args.beta_s = 1.0f;    // Beta already applied to C
    st_args.thread_id = 0;    // Main thread uses buffer slot 0
    
    for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
    {
        pb = MIN(k - p, cblas_gemm_kc);
        for (CBLAS_INDEX row = 0; row < m; row += cblas_gemm_mc) 
        {
            ib = MIN(m - row, cblas_gemm_mc);
            
            // Setup args for this tile
            st_args.a = &A(p, row);
            st_args.b = &B(0, p);
            st_args.c = &C(0, row);
            st_args.ib = ib;
            st_args.pb = pb;
            
            // Dispatch to platform-optimized kernel
            blas_kernels.sgemm_k(&st_args);
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
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

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

#elif defined(__aarch64__) && defined(__ARM_NEON)

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

    // Use cache-blocked kernel-based implementation
    CBLAS_INDEX pb, ib;
    cblas_args_t st_args;
    
    st_args.n = n;
    st_args.lda = lda;
    st_args.ldb = ldb;
    st_args.ldc = ldc;
    st_args.alpha_d = alpha;  // Pass alpha to kernel
    st_args.beta_d = 1.0;     // Beta already applied to C
    st_args.thread_id = 0;    // Main thread uses buffer slot 0
    
    // Double precision uses smaller tiles (half elements per cache line)
    // Use mc/2, kc, nb for double precision
    CBLAS_INDEX mc_d = cblas_gemm_mc;
    CBLAS_INDEX kc_d = cblas_gemm_kc;
    
    for (CBLAS_INDEX p = 0; p < k; p += kc_d) 
    {
        pb = MIN(k - p, kc_d);
        for (CBLAS_INDEX row = 0; row < m; row += mc_d) 
        {
            ib = MIN(m - row, mc_d);
            
            // Setup args for this tile
            st_args.a = &A(p, row);
            st_args.b = &B(0, p);
            st_args.c = &C(0, row);
            st_args.ib = ib;
            st_args.pb = pb;
            
            // Dispatch to platform-optimized kernel
            blas_kernels.dgemm_k(&st_args);
        }
    }

    CBLAS_STATS_END("dgemm", m * n * k, mt_used);
}

