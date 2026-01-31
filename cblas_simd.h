//------------------------------------------------------
// cblas_simd.h
//
// Platform-specific SIMD intrinsic headers for CBLAS
// Include this header in kernel files that use SIMD optimizations
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#ifndef __CBLAS_SIMD_H
#define __CBLAS_SIMD_H

// X86/X64 SIMD intrinsics (SSE, AVX, AVX2, FMA)
#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#   include <immintrin.h>
#endif

// ARM NEON intrinsics
#if defined(__aarch64__) && defined(__ARM_NEON)
#   include <arm_neon.h>
#endif

#endif // __CBLAS_SIMD_H
