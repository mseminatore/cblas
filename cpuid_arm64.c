//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <stdint.h>
#include "cblas.h"
#include "kernels.h"

#if !defined(_WIN32)
#   include <unistd.h>
#endif

#ifdef __APPLE__
#   include <sys/sysctl.h>
//#include <mach/machine.h>
#endif

enum {
    CPU_GENERIC,
    CPU_VORTEX_M1,
    CPU_VORTEX_M2,
    CPU_VORTEX_M3
};

static const char *cpu_names[] =
{
    "Generic ARM Core",
    "Vortex-M1",
    "Vortex-M2",
    "Vortex-M3"
};

static unsigned int cpu_features = CPU_NONE;

//------------------------------------------------------
// return the name of the current processor
//------------------------------------------------------
const char *cpu_get_core_name(void)
{
#ifdef __APPLE__
    uint32_t entry;
    size_t len = sizeof(entry);
    sysctlbyname("hw.cpufamily", &entry, &len, NULL, 0);

    // these values taken from <mach/machine.h>
    if (entry == 131287967 || entry == 458787763)
        return cpu_names[CPU_VORTEX_M1];
    else if (entry == 3660830781)
        return cpu_names[CPU_VORTEX_M2];
    else if (entry == 2271604202)
        return cpu_names[CPU_VORTEX_M3];
    else
        return cpu_names[CPU_GENERIC];

#endif
    return cpu_names[CPU_GENERIC];
}

//------------------------------------------------------
// query and return any ISA features of this CPU
//------------------------------------------------------
static unsigned int __cpu_get_features(void)
{
#if defined(__APPLE__) || defined(__aarch64__)
    cpu_features |= CPU_NEON;
    cpu_features |= CPU_NEON_FMA;
#endif

    // Initialize Level-1 kernel function pointers
    blas_kernels.sdot_k = cblas_sdot_k;
    blas_kernels.ddot_k = cblas_ddot_k;
    blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc;
    blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc;

    blas_kernels.sasum_k = cblas_sasum_k;
    blas_kernels.dasum_k = cblas_dasum_k;
    blas_kernels.sasum_k_noinc = cblas_sasum_k_noinc;
    blas_kernels.dasum_k_noinc = cblas_dasum_k_noinc;

	blas_kernels.scopy_k = cblas_scopy_k;
	blas_kernels.dcopy_k = cblas_dcopy_k;
	blas_kernels.scopy_k_noinc = cblas_scopy_k_noinc;
	blas_kernels.dcopy_k_noinc = cblas_dcopy_k_noinc;

	// Initialize Level-2 kernel function pointers
	blas_kernels.sger_k = sger_k;
	blas_kernels.dger_k = dger_k;
	blas_kernels.sgemv_k = sgemv_k;
	blas_kernels.dgemv_k = dgemv_k;

	// Initialize Level-3 kernel function pointers
	blas_kernels.sgemm_k = sgemm_k;

    // use NEON optimized kernels if available
	if (cpu_features & CPU_NEON)
	{
        blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc_neon;  // NEON optimized version
        blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc_neon;  // NEON optimized version
        blas_kernels.sasum_k_noinc = cblas_sasum_k_noinc_neon;
        blas_kernels.dasum_k_noinc = cblas_dasum_k_noinc_neon;
        blas_kernels.scopy_k_noinc = cblas_scopy_k_noinc_neon;
        blas_kernels.dcopy_k_noinc = cblas_dcopy_k_noinc_neon;
    }

    return cpu_features;
}

#if !defined(_WIN32)
#include <pthread.h>
static pthread_mutex_t cpu_features_lock = PTHREAD_MUTEX_INITIALIZER;
#endif

//------------------------------------------------------
// get and cache cpu features
//------------------------------------------------------
unsigned int cpu_get_features(void)
{
	// Fast path: return cached value if already initialized
	if (cpu_features != CPU_NONE)
		return cpu_features;

#if !defined(_WIN32)
	// Slow path: need to initialize (double-checked locking)
	pthread_mutex_lock(&cpu_features_lock);
	// Check again after acquiring lock (another thread may have initialized)
	if (cpu_features == CPU_NONE)
		cpu_features = __cpu_get_features();
	pthread_mutex_unlock(&cpu_features_lock);
#else
	if (cpu_features == CPU_NONE)
		cpu_features = __cpu_get_features();
#endif

	return cpu_features;
}

//------------------------------------------------------
// return the number of CPU cores available
//------------------------------------------------------
int cpu_get_core_count(void)
{    
#ifdef __APPLE__
    uint32_t entry;
    size_t len = sizeof(entry);

    // TODO - per sysctl.h this might want to be hw.ncpu?
    sysctlbyname("hw.physicalcpu_max", &entry, &len, NULL, 0);
    return entry;
#endif

#ifndef _WIN32
    int cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cores == -1)
        cores = 1;

    return cores;
#endif

    // catch all
    puts("warning: cpu_get_core_count() not implemented!");
    return 1;
}

//------------------------------------------------------
// return the CPU L1 cache line size
//------------------------------------------------------
int cpu_get_cacheline_size(void)
{
#ifdef __APPLE__
    long int entry;
    size_t len = sizeof(entry);

    sysctlbyname("hw.cachelinesize", &entry, &len, NULL, 0);
    return entry;
#else

    long l1 = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    return l1;

#endif

    // catch all
    puts("warning: cpu_get_cacheline_size() not implemented!");
    return 32;
}

//------------------------------------------------------
// return the L2$ size
//------------------------------------------------------
int cpu_get_l2_cache_size(void)
{
    long l2_cache_size = 0;

#ifdef __APPLE__
    size_t len = sizeof(l2_cache_size);
    sysctlbyname("hw.l2cachesize", &l2_cache_size, &len, NULL, 0);
#else
    l2_cache_size = sysconf(_SC_LEVEL2_CACHE_SIZE);

#endif

    return l2_cache_size/1024;
}

//------------------------------------------------------
// return the L1 data cache size in KBytes
//------------------------------------------------------
int cpu_get_l1_data_cache_size(void)
{
    long l1_cache_size = 0;

#ifdef __APPLE__
    size_t len = sizeof(l1_cache_size);
    
    if (sysctlbyname("hw.l1dcachesize", &l1_cache_size, &len, NULL, 0) != 0)
        return 0;  // Return 0 on failure to trigger fallback
    return l1_cache_size/1024;
#else
    l1_cache_size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
    if (l1_cache_size <= 0)
        return 0;  // Return 0 on failure to trigger fallback
    return l1_cache_size/1024;
#endif
}

//------------------------------------------------------
// Check if CPU has hybrid architecture (P-cores + E-cores)
// ARM64 doesn't have hybrid architectures like x86
//------------------------------------------------------
int cpu_is_hybrid(void)
{
    return 0;  // ARM64 systems don't have hybrid P/E core architectures
}

//------------------------------------------------------
// Get number of P-cores (performance cores)
// Not applicable for ARM64
//------------------------------------------------------
int cpu_get_p_core_count(void)
{
    return 0;  // Not applicable for ARM64
}

//------------------------------------------------------
// Get number of E-cores (efficiency cores)
// Not applicable for ARM64
//------------------------------------------------------
int cpu_get_e_core_count(void)
{
    return 0;  // Not applicable for ARM64
}
