//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#include "cblas.h"
#include "cblas_simd.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)

#define GEMV_SUFFIX fma
#define GEMV_MADD_PS(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#define GEMV_MADD_PD(a, b, c) _mm256_fmadd_pd((a), (b), (c))

#include "gemv_k_x86.h"

#endif // x86/x64
