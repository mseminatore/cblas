//------------------------------------------------------
// platform/cpuid.h
//
// Platform-agnostic CPU detection and feature query interface
// Abstracts x86_64 CPUID and ARM64 CPU feature detection
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------

#ifndef __PLATFORM_CPUID_H
#define __PLATFORM_CPUID_H

/**
 * @brief Get the number of CPU cores
 * @return Number of CPU cores available
 */
int cpu_get_core_count(void);

/**
 * @brief Get the CPU manufacturer/model name
 * @return String describing the CPU
 */
const char *cpu_get_core_name(void);

/**
 * @brief Get the CPU L1 cache line size
 * @return Cache line size in bytes
 */
int cpu_get_cacheline_size(void);

/**
 * @brief Get the CPU L2 cache size
 * @return L2 cache size in KB
 */
int cpu_get_l2_cache_size(void);

/**
 * @brief Get the CPU L1 data cache size
 * @return L1 data cache size in KB
 */
int cpu_get_l1_data_cache_size(void);

/**
 * @brief Query CPU feature flags (ISA extensions)
 * @return Bitmask of CPU_* feature flags
 * @note This function is cached - first call does detection, subsequent calls return cached value
 */
unsigned int cpu_get_features(void);

/**
 * @brief Check if CPU has hybrid architecture (P-cores + E-cores)
 * @return 1 if hybrid, 0 otherwise
 * @note Only x86_64 Intel Alder Lake and newer support hybrid architectures
 */
int cpu_is_hybrid(void);

/**
 * @brief Get number of P-cores (performance cores)
 * @return Number of P-cores, or 0 if not a hybrid CPU or not supported
 */
int cpu_get_p_core_count(void);

/**
 * @brief Get number of E-cores (efficiency cores)
 * @return Number of E-cores, or 0 if not a hybrid CPU or not supported
 */
int cpu_get_e_core_count(void);

#endif // __PLATFORM_CPUID_H
