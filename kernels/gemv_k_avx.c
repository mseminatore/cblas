//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

// Compiled with -mavx (not -mavx2): only 256-bit float ops are needed, so
// Sandy Bridge / Ivy Bridge get a vector kernel instead of the scalar base.
#if (defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)) && defined(__AVX__)

#define GEMV_SUFFIX avx
#define GEMV_MADD_PS(a, b, c) _mm256_add_ps((c), _mm256_mul_ps((a), (b)))
#define GEMV_MADD_PD(a, b, c) _mm256_add_pd((c), _mm256_mul_pd((a), (b)))

#include "gemv_k_x86.h"

#endif // x86/x64 && AVX
