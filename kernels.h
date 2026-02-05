//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#ifndef __KERNELS_H
#define __KERNELS_H

// Level-1 kernel function declarations
void cblas_sdot_k(cblas_args_t* args);
void cblas_sdot_k_noinc(cblas_args_t* args);
void cblas_sdot_k_noinc_neon(cblas_args_t* args);
void cblas_sdot_k_noinc_sse(cblas_args_t* args);
void cblas_sdot_k_noinc_avx(cblas_args_t* args);
void cblas_sdot_k_noinc_fma(cblas_args_t* args);
void cblas_sdot_k_noinc_avx512(cblas_args_t* args);

void cblas_ddot_k(cblas_args_t* args);
void cblas_ddot_k_noinc(cblas_args_t* args);
void cblas_ddot_k_noinc_neon(cblas_args_t* args);
void cblas_ddot_k_noinc_sse(cblas_args_t* args);
void cblas_ddot_k_noinc_avx(cblas_args_t* args);
void cblas_ddot_k_noinc_fma(cblas_args_t* args);
void cblas_ddot_k_noinc_avx512(cblas_args_t* args);

void cblas_sasum_k(cblas_args_t* args);
void cblas_dasum_k(cblas_args_t* args);
void cblas_sasum_k_noinc(cblas_args_t* args);
void cblas_dasum_k_noinc(cblas_args_t* args);
void cblas_sasum_k_noinc_sse(cblas_args_t* args);
void cblas_dasum_k_noinc_sse(cblas_args_t* args);
void cblas_sasum_k_noinc_avx(cblas_args_t* args);
void cblas_dasum_k_noinc_avx(cblas_args_t* args);
void cblas_sasum_k_noinc_neon(cblas_args_t* args);
void cblas_dasum_k_noinc_neon(cblas_args_t* args);

void cblas_scopy_k(cblas_args_t* args);
void cblas_dcopy_k(cblas_args_t* args);
void cblas_scopy_k_noinc(cblas_args_t* args);
void cblas_dcopy_k_noinc(cblas_args_t* args);
void cblas_scopy_k_noinc_sse(cblas_args_t* args);
void cblas_dcopy_k_noinc_sse(cblas_args_t* args);
void cblas_scopy_k_noinc_avx(cblas_args_t* args);
void cblas_dcopy_k_noinc_avx(cblas_args_t* args);
void cblas_scopy_k_noinc_neon(cblas_args_t* args);
void cblas_dcopy_k_noinc_neon(cblas_args_t* args);

void cblas_sswap_k(cblas_args_t* args);
void cblas_sswap_k_noinc(cblas_args_t* args);
void cblas_sswap_k_noinc_sse(cblas_args_t* args);
void cblas_sswap_k_noinc_neon(cblas_args_t* args);
void cblas_dswap_k(cblas_args_t* args);
void cblas_dswap_k_noinc(cblas_args_t* args);
void cblas_dswap_k_noinc_sse(cblas_args_t* args);
void cblas_dswap_k_noinc_neon(cblas_args_t* args);

// Level-2 kernel function declarations

// Level-3 kernel function declarations

#endif // __KERNELS_H

