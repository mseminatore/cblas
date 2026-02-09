//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

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
    // Our kernels use row-major storage: A(col, row) = a[row * lda + col]
    // 
    // For CblasRowMajor: User stores row-major, which matches our kernels directly.
    //   A is m×k with A[i,j] = a[i*lda+j], B is k×n with B[i,j] = b[i*ldb+j]
    //   This matches kernel expectation, no transformation needed.
    //
    // For CblasColMajor: User stores column-major, but our kernels expect row-major.
    //   Apply the identity: C = alpha * op(A) * op(B) + beta * C in col-major
    //   is equivalent to: C^T = alpha * op(B)^T * op(A)^T + beta * C^T in row-major
    //   So we swap A↔B, m↔n, lda↔ldb, and swap transpose flags.
    
    float *a_use, *b_use;
    CBLAS_INDEX m_use, n_use, lda_use, ldb_use;
    CBLAS_TRANSPOSE transa_use, transb_use;
    
    if (layout == CblasColMajor)
    {
        // Column-major input: swap to convert to row-major computation
        a_use = b;
        b_use = a;
        m_use = n;
        n_use = m;
        lda_use = ldb;
        ldb_use = lda;
        transa_use = transb;
        transb_use = transa;
    }
    else
    {
        // Row-major: use as-is (matches our kernel convention)
        a_use = a;
        b_use = b;
        m_use = m;
        n_use = n;
        lda_use = lda;
        ldb_use = ldb;
        transa_use = transa;
        transb_use = transb;
    }

#ifdef CBLAS_CHECK_INPUTS
    CBLAS_INDEX nota = (transa_use == CblasNoTrans);
    CBLAS_INDEX notb = (transb_use == CblasNoTrans);

    // Row-major storage: lda must be >= number of columns of A
    // NoTrans: A is m×k, so lda >= k
    // Trans: A is k×m (accessed as m×k transposed), so lda >= m
    CBLAS_INDEX ncola = nota ? k : m_use;
    CBLAS_INDEX ncolb = notb ? n_use : k;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (!a_use)
        info = 8;
    else if (lda_use < MAX(1, ncola))
        info = 9;
    else if (!b_use)
        info = 10;
    else if (ldb_use < MAX(1, ncolb))
        info = 11;
    else if (!c)
        info = 12;
    else if (ldc < MAX(1, n_use))
        info = 13;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (m_use < 0 || n_use < 0 || k < 0 || !a_use || !b_use || !c || 
        lda_use < MAX(1, ncola) || ldb_use < MAX(1, ncolb) || ldc < MAX(1, n_use))
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
    // C has dimensions m_use x n_use in row-major: C[i,j] = c[i * ldc + j]
    if (beta == 0.0f)
    {
        for (CBLAS_INDEX i = 0; i < m_use; i++)
            for (CBLAS_INDEX j = 0; j < n_use; j++)
                c[i * ldc + j] = 0.0f;
    }
    else if (beta != 1.0f)
    {
        for (CBLAS_INDEX i = 0; i < m_use; i++)
            for (CBLAS_INDEX j = 0; j < n_use; j++)
                c[i * ldc + j] *= beta;
    }

    if (alpha == 0.0f)
    {
        CBLAS_STATS_END("sgemm", m * n * k, 0);
        return;
    }

    // Check for transpose - currently only NoTrans/NoTrans is optimized
    if (transa_use != CblasNoTrans || transb_use != CblasNoTrans)
    {
        // Reference implementation with transpose support
        // Row-major: C[i,j] = c[i * ldc + j]
        // C = alpha * op(A) * op(B) + C  (beta already applied)
        for (CBLAS_INDEX i = 0; i < m_use; i++)
        {
            for (CBLAS_INDEX j = 0; j < n_use; j++)
            {
                float sum = 0.0f;
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    // A is m_use x k (notrans) or k x m_use (trans)
                    // Row-major: A[i,p] = a[i*lda+p] (notrans), A^T[i,p] = a[p*lda+i] (trans)
                    float a_val = (transa_use == CblasNoTrans) 
                        ? a_use[i * lda_use + p]
                        : a_use[p * lda_use + i];
                    
                    // B is k x n_use (notrans) or n_use x k (trans)
                    // Row-major: B[p,j] = b[p*ldb+j] (notrans), B^T[p,j] = b[j*ldb+p] (trans)
                    float b_val = (transb_use == CblasNoTrans)
                        ? b_use[p * ldb_use + j]
                        : b_use[j * ldb_use + p];
                    
                    sum += a_val * b_val;
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
        CBLAS_STATS_END("sgemm", m * n * k, 0);
        return;
    }

    // NoTrans/NoTrans case - use optimized kernels
    // The kernels use row-major convention: A(col, row) = a[row * lda + col]
    
    CBLAS_INDEX pb, ib;

#if defined(MT_ENABLED)
    int mt_used = (m_use * n_use * k > CBLAS_MT_GEMM) ? 1 : 0;
    
    if (mt_used)
    {
        CBLAS_INDEX num_threads = cblas_get_num_threads();
        
        // For small matrices, use smaller tile sizes to improve parallelism
        // Goal: have at least num_threads tiles for good load balancing
        CBLAS_INDEX mc_use = cblas_gemm_mc;
        CBLAS_INDEX nb_use = cblas_gemm_nb;
        
        // Calculate initial tile counts
        CBLAS_INDEX vert_tiles = (m_use + mc_use - 1) / mc_use;
        CBLAS_INDEX horiz_tiles = (n_use + nb_use - 1) / nb_use;
        
        // If we have too few tiles for good parallelism, use adaptive tiling
        // Threshold: need at least num_threads/2 tiles for reasonable load balance
        CBLAS_INDEX min_tiles_for_mt = (num_threads + 1) / 2;
        if (vert_tiles * horiz_tiles < min_tiles_for_mt && m_use >= 256 && n_use >= 256)
        {
            // Use smaller tiles to create more parallel work
            // Target: at least min_tiles_for_mt total tiles
            CBLAS_INDEX target_tiles_per_dim = (CBLAS_INDEX)(sqrt((double)min_tiles_for_mt) + 0.5);
            if (target_tiles_per_dim < 2) target_tiles_per_dim = 2;
            
            mc_use = (m_use + target_tiles_per_dim - 1) / target_tiles_per_dim;
            nb_use = (n_use + target_tiles_per_dim - 1) / target_tiles_per_dim;
            
            // Ensure minimum tile size for kernel efficiency
            CBLAS_INDEX min_tile = 64;
            mc_use = MAX(mc_use, min_tile);
            nb_use = MAX(nb_use, min_tile);
            
            // Recalculate tile counts
            vert_tiles = (m_use + mc_use - 1) / mc_use;
            horiz_tiles = (n_use + nb_use - 1) / nb_use;
        }
        
        CBLAS_INDEX total_tiles = vert_tiles * horiz_tiles;
        int use_2d_tiling = (horiz_tiles > 1);

        work_queue_t *queue = (work_queue_t*)malloc(total_tiles * sizeof(work_queue_t));
        cblas_args_t *args = (cblas_args_t*)malloc(total_tiles * sizeof(cblas_args_t));
        
        if (!queue || !args) {
            free(queue);
            free(args);
            return;
        }

        for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
        {
            pb = MIN(k - p, cblas_gemm_kc);
            CBLAS_INDEX tile_count = 0;
            
            if (use_2d_tiling)
            {
                // 2D tiling: parallelize on both rows and columns
                for (CBLAS_INDEX row = 0; row < m_use; row += mc_use) 
                {
                    ib = MIN(m_use - row, mc_use);
                    
                    for (CBLAS_INDEX col = 0; col < n_use; col += nb_use)
                    {
                        CBLAS_INDEX jb = MIN(n_use - col, nb_use);
                        
                        args[tile_count].incx = 1;
                        args[tile_count].incy = 1;
                        args[tile_count].n = jb;  // Only process jb columns
                        args[tile_count].lda = lda_use;
                        args[tile_count].ldb = ldb_use;
                        args[tile_count].ldc = ldc;
                        args[tile_count].a = a_use + row * lda_use + p;
                        args[tile_count].b = b_use + p * ldb_use + col;  // B offset by col
                        args[tile_count].c = c + row * ldc + col;        // C offset by col
                        args[tile_count].ib = ib;
                        args[tile_count].pb = pb;
                        args[tile_count].alpha_s = alpha;
                        args[tile_count].beta_s = 1.0f;
                        args[tile_count].thread_id = 0;
                    
                        queue[tile_count].finished = 0;
                        queue[tile_count].args = &args[tile_count];
                        queue[tile_count].kernel = blas_kernels.sgemm_k;
                        queue[tile_count].next = &queue[tile_count + 1];

                        tile_count++;
                    }
                }
            }
            else
            {
                // 1D tiling: parallelize on rows only (original behavior)
                for (CBLAS_INDEX row = 0; row < m_use; row += mc_use) 
                {
                    ib = MIN(m_use - row, mc_use);

                    args[tile_count].incx = 1;
                    args[tile_count].incy = 1;
                    args[tile_count].n = n_use;
                    args[tile_count].lda = lda_use;
                    args[tile_count].ldb = ldb_use;
                    args[tile_count].ldc = ldc;
                    args[tile_count].a = a_use + row * lda_use + p;
                    args[tile_count].b = b_use + p * ldb_use;
                    args[tile_count].c = c + row * ldc;
                    args[tile_count].ib = ib;
                    args[tile_count].pb = pb;
                    args[tile_count].alpha_s = alpha;
                    args[tile_count].beta_s = 1.0f;
                    args[tile_count].thread_id = 0;
                
                    queue[tile_count].finished = 0;
                    queue[tile_count].args = &args[tile_count];
                    queue[tile_count].kernel = blas_kernels.sgemm_k;
                    queue[tile_count].next = &queue[tile_count + 1];

                    tile_count++;
                }
            }

            assert(tile_count <= total_tiles);

            queue[tile_count - 1].next = NULL;

            cblas_execute(tile_count, queue);
        }
        
        free(queue);
        free(args);
    }
    else
    {
        // Below threshold, use single-threaded implementation
        cblas_args_t st_args;
        st_args.incx = 1;
        st_args.incy = 1;
        st_args.n = n_use;
        st_args.lda = lda_use;
        st_args.ldb = ldb_use;
        st_args.ldc = ldc;
        st_args.alpha_s = alpha;
        st_args.beta_s = 1.0f;
        st_args.thread_id = 0;
        
        for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
        {
            pb = MIN(k - p, cblas_gemm_kc);
            
            for (CBLAS_INDEX row = 0; row < m_use; row += cblas_gemm_mc) 
            {
                ib = MIN(m_use - row, cblas_gemm_mc);
                
                // Kernel uses A(col, row) = a[row * lda + col]
                st_args.a = a_use + row * lda_use + p;  // &A(p, row)
                st_args.b = b_use + p * ldb_use;        // &B(0, p)
                st_args.c = c + row * ldc;              // &C(0, row)
                st_args.ib = ib;
                st_args.pb = pb;
                
                blas_kernels.sgemm_k(&st_args);
            }
        }
    }

#else
    int mt_used = 0;
    // Single-threaded implementation
    cblas_args_t st_args;
    st_args.incx = 1;
    st_args.incy = 1;
    st_args.n = n_use;
    st_args.lda = lda_use;
    st_args.ldb = ldb_use;
    st_args.ldc = ldc;
    st_args.alpha_s = alpha;
    st_args.beta_s = 1.0f;
    st_args.thread_id = 0;
    
    for (CBLAS_INDEX p = 0; p < k; p += cblas_gemm_kc) 
    {
        pb = MIN(k - p, cblas_gemm_kc);
        for (CBLAS_INDEX row = 0; row < m_use; row += cblas_gemm_mc) 
        {
            ib = MIN(m_use - row, cblas_gemm_mc);
            
            st_args.a = a_use + row * lda_use + p;
            st_args.b = b_use + p * ldb_use;
            st_args.c = c + row * ldc;
            st_args.ib = ib;
            st_args.pb = pb;
            
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

        // Rows 1-2 using SSE (non-FMA)
        c_row1 = _mm_add_pd(c_row1, _mm_mul_pd(a_p0, b_row));
        c_row2 = _mm_add_pd(c_row2, _mm_mul_pd(a_p1, b_row));
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
    // Our kernels use row-major storage: A(col, row) = a[row * lda + col]
    // 
    // For CblasRowMajor: User stores row-major, which matches our kernels directly.
    // For CblasColMajor: Apply swap transformation to convert to row-major.
    
    double *a_use, *b_use;
    CBLAS_INDEX m_use, n_use, lda_use, ldb_use;
    CBLAS_TRANSPOSE transa_use, transb_use;
    
    if (layout == CblasColMajor)
    {
        // Column-major input: swap to convert to row-major computation
        a_use = b;
        b_use = a;
        m_use = n;
        n_use = m;
        lda_use = ldb;
        ldb_use = lda;
        transa_use = transb;
        transb_use = transa;
    }
    else
    {
        // Row-major: use as-is (matches our kernel convention)
        a_use = a;
        b_use = b;
        m_use = m;
        n_use = n;
        lda_use = lda;
        ldb_use = ldb;
        transa_use = transa;
        transb_use = transb;
    }

#ifdef CBLAS_CHECK_INPUTS
    CBLAS_INDEX nota = (transa_use == CblasNoTrans);
    CBLAS_INDEX notb = (transb_use == CblasNoTrans);

    // Row-major storage: lda must be >= number of columns of A
    CBLAS_INDEX ncola = nota ? k : m_use;
    CBLAS_INDEX ncolb = notb ? n_use : k;

#ifdef CBLAS_XERBLA_INPUTS
    int info = 0;
    if (!a_use)
        info = 8;
    else if (lda_use < MAX(1, ncola))
        info = 9;
    else if (!b_use)
        info = 10;
    else if (ldb_use < MAX(1, ncolb))
        info = 11;
    else if (!c)
        info = 12;
    else if (ldc < MAX(1, n_use))
        info = 13;

    if (info) {
        XERBLA(info);
        return;
    }
#else
    if (m_use < 0 || n_use < 0 || k < 0 || !a_use || !b_use || !c ||
        lda_use < MAX(1, ncola) || ldb_use < MAX(1, ncolb) || ldc < MAX(1, n_use))
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

    int mt_used = 0;

    // Handle beta scaling of C (row-major: C[i,j] = c[i * ldc + j])
    if (beta == 0.0)
    {
        for (CBLAS_INDEX i = 0; i < m_use; i++)
            for (CBLAS_INDEX j = 0; j < n_use; j++)
                c[i * ldc + j] = 0.0;
    }
    else if (beta != 1.0)
    {
        for (CBLAS_INDEX i = 0; i < m_use; i++)
            for (CBLAS_INDEX j = 0; j < n_use; j++)
                c[i * ldc + j] *= beta;
    }

    if (alpha == 0.0)
    {
        CBLAS_STATS_END("dgemm", m * n * k, mt_used);
        return;
    }

    // Check for transpose - fall back to reference implementation
    if (transa_use != CblasNoTrans || transb_use != CblasNoTrans)
    {
        // Row-major reference: C[i,j] = c[i * ldc + j]
        for (CBLAS_INDEX i = 0; i < m_use; i++)
        {
            for (CBLAS_INDEX j = 0; j < n_use; j++)
            {
                double sum = 0.0;
                for (CBLAS_INDEX p = 0; p < k; p++)
                {
                    double a_val = (transa_use == CblasNoTrans) 
                        ? a_use[i * lda_use + p]
                        : a_use[p * lda_use + i];
                    
                    double b_val = (transb_use == CblasNoTrans)
                        ? b_use[p * ldb_use + j]
                        : b_use[j * ldb_use + p];
                    
                    sum += a_val * b_val;
                }
                c[i * ldc + j] += alpha * sum;
            }
        }
        CBLAS_STATS_END("dgemm", m * n * k, mt_used);
        return;
    }

    // NoTrans/NoTrans case - use optimized kernels
    // Kernels use row-major: A(col, row) = a[row * lda + col]
    CBLAS_INDEX pb, ib;
    cblas_args_t st_args;
    
    st_args.n = n_use;
    st_args.lda = lda_use;
    st_args.ldb = ldb_use;
    st_args.ldc = ldc;
    st_args.alpha_d = alpha;
    st_args.beta_d = 1.0;
    st_args.thread_id = 0;
    
    CBLAS_INDEX mc_d = cblas_gemm_mc;
    CBLAS_INDEX kc_d = cblas_gemm_kc;
    
    for (CBLAS_INDEX p = 0; p < k; p += kc_d) 
    {
        pb = MIN(k - p, kc_d);
        for (CBLAS_INDEX row = 0; row < m_use; row += mc_d) 
        {
            ib = MIN(m_use - row, mc_d);
            
            // Row-major kernel: a + row * lda + p = &A(p, row)
            st_args.a = a_use + row * lda_use + p;
            st_args.b = b_use + p * ldb_use;
            st_args.c = c + row * ldc;
            st_args.ib = ib;
            st_args.pb = pb;
            
            blas_kernels.dgemm_k(&st_args);
        }
    }

    CBLAS_STATS_END("dgemm", m * n * k, mt_used);
}

