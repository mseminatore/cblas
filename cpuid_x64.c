//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cblas.h"
#include "kernels.h"

#ifdef _WIN32
#	include <Windows.h>
#else
#   include <unistd.h>
#endif

#if defined(__clang__) || defined(__GNUC__)
#	include <cpuid.h>
#endif

#ifdef __APPLE__
#	include <sys/sysctl.h>
#endif

#define EAX 0
#define EBX 1
#define ECX 2
#define EDX 3

static unsigned int cpu_features = CPU_NONE;

#define BIT(i) (1 << (i))

//------------------------------------------------------
// return the CPU name
//------------------------------------------------------
const char *cpu_get_core_name(void)
{
	static char mfgID[13];
#if defined(_MSC_VER)
	int info[4];

	__cpuid(info, 0);

	((int*)mfgID)[0] = info[1];
	((int*)mfgID)[1] = info[3];
	((int*)mfgID)[2] = info[2];
	mfgID[12] = 0;

	return mfgID;
#else
	unsigned int eax, ebx, ecx, edx;

	__cpuid(0, eax, ebx, ecx, edx);
	((int*)mfgID)[0] = ebx;
	((int*)mfgID)[1] = edx;
	((int*)mfgID)[2] = ecx;
	mfgID[12] = 0;

	return mfgID;

	return "Generic x64";
#endif
}

//------------------------------------------------------
// return the CPU L1 cache line size
//------------------------------------------------------
int cpu_get_cacheline_size(void)
{
	int line_size = 64;

#ifdef __APPLE__
	long entry;
	size_t len = sizeof(entry);

	sysctlbyname("hw.cachelinesize", &entry, &len, NULL, 0);
	return entry;
#else

#if defined(_MSC_VER)
	int regs[4];
	__cpuid(regs, 0x80000006);
	line_size = regs[ECX] & 0xff;
#else
	unsigned int eax, ebx, ecx, edx;
	__cpuid(0x80000006, eax, ebx, ecx, edx);
	line_size = ecx & 0xff;	
#endif

#endif

	return line_size;
}

//------------------------------------------------------
// return the L2$ size in KBytes
//------------------------------------------------------
int cpu_get_l2_cache_size(void)
{
    long l2_cache_size = 0;

#ifdef __APPLE__
    size_t len = sizeof(l2_cache_size);

    sysctlbyname("hw.l2cachesize", &l2_cache_size, &len, NULL, 0);
    return l2_cache_size/1024;
#else
	#if defined(_MSC_VER)
		int regs[4];
		__cpuid(regs, 0x80000006);
		l2_cache_size = (regs[ECX] >> 16) & 0xFFFF; // Extract L2 cache size in KB
	#else
		unsigned int eax, ebx, ecx, edx = 0;

		__cpuid(0x80000006, eax, ebx, ecx, edx);
		l2_cache_size = (ecx >> 16) & 0xFFFF; // Extract L2 cache size in KB
	#endif
#endif

	return l2_cache_size;
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
	// Get vendor string to determine Intel vs AMD
	const char* vendor = cpu_get_core_name();
	
	// Intel uses CPUID leaf 0x04 (Deterministic Cache Parameters)
	if (strncmp(vendor, "GenuineIntel", 12) == 0)
	{
		#if defined(_MSC_VER)
			// Iterate through cache levels to find L1 data cache
			for (int i = 0; i < 32; i++)
			{
				int regs[4];
				__cpuidex(regs, 0x04, i);
				
				int cache_type = regs[EAX] & 0x1F;
				if (cache_type == 0) break; // No more caches
				
				int cache_level = (regs[EAX] >> 5) & 0x7;
				
				// cache_type: 1=Data, 2=Instruction, 3=Unified
				// We want L1 (level 1) Data cache (type 1)
				if (cache_level == 1 && cache_type == 1)
				{
					int line_size = (regs[EBX] & 0xFFF) + 1;
					int partitions = ((regs[EBX] >> 12) & 0x3FF) + 1;
					int associativity = ((regs[EBX] >> 22) & 0x3FF) + 1;
					int sets = regs[ECX] + 1;
					
					l1_cache_size = (line_size * partitions * associativity * sets) / 1024;
					break;
				}
			}
		#else
			// GCC/Clang version
			for (int i = 0; i < 32; i++)
			{
				unsigned int eax, ebx, ecx, edx;
				__cpuid_count(0x04, i, eax, ebx, ecx, edx);
				
				int cache_type = eax & 0x1F;
				if (cache_type == 0) break;
				
				int cache_level = (eax >> 5) & 0x7;
				
				if (cache_level == 1 && cache_type == 1)
				{
					int line_size = (ebx & 0xFFF) + 1;
					int partitions = ((ebx >> 12) & 0x3FF) + 1;
					int associativity = ((ebx >> 22) & 0x3FF) + 1;
					int sets = ecx + 1;
					
					l1_cache_size = (line_size * partitions * associativity * sets) / 1024;
					break;
				}
			}
		#endif
	}
	// AMD uses CPUID leaf 0x80000005
	else if (strncmp(vendor, "AuthenticAMD", 12) == 0)
	{
		#if defined(_MSC_VER)
			int regs[4];
			__cpuid(regs, 0x80000005);
			l1_cache_size = (regs[ECX] >> 24) & 0xFF; // Extract L1 data cache size in KB
		#else
			unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
			__cpuid(0x80000005, eax, ebx, ecx, edx);
			l1_cache_size = (ecx >> 24) & 0xFF; // Extract L1 data cache size in KB
		#endif
	}
	else
	{
		// Unknown vendor, try AMD method as fallback
		#if defined(_MSC_VER)
			int regs[4];
			__cpuid(regs, 0x80000005);
			l1_cache_size = (regs[ECX] >> 24) & 0xFF;
		#else
			unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
			__cpuid(0x80000005, eax, ebx, ecx, edx);
			l1_cache_size = (ecx >> 24) & 0xFF;
		#endif
	}
#endif

	return l1_cache_size;
}

//------------------------------------------------------
// return the CPU core brand name
//------------------------------------------------------
const char* cpu_get_brand_string(void)
{
#if defined(_MSC_VER)
	int regs[12];
	static char str[sizeof(regs)];

	__cpuid(regs, 0x80000000);

	if (regs[0] < 0x80000004)
		return "No brand string";

	__cpuid(&regs[0], 0x80000002);
	__cpuid(&regs[4], 0x80000003);
	__cpuid(&regs[8], 0x80000004);

	memcpy(str, regs, sizeof(regs));
	return str;
#else
	return "Generic x64";
#endif
}

//------------------------------------------------------
// initialize BLAS kernel function pointers
//------------------------------------------------------
static void init_blas_kernels()
{
    // Initialize Level-1 kernel function pointers
    blas_kernels.sdot_k = cblas_sdot_k;
    blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc;
    blas_kernels.ddot_k = cblas_ddot_k;
    blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc;
	blas_kernels.sasum_k = cblas_sasum_k;
	blas_kernels.dasum_k = cblas_dasum_k;
	blas_kernels.sasum_k_noinc = cblas_sasum_k_noinc;
	blas_kernels.dasum_k_noinc = cblas_dasum_k_noinc;

	blas_kernels.scopy_k = cblas_scopy_k;
	blas_kernels.dcopy_k = cblas_dcopy_k;
	blas_kernels.scopy_k_noinc = cblas_scopy_k_noinc;
	blas_kernels.dcopy_k_noinc = cblas_dcopy_k_noinc;

	//blas_kernels.sscal_k = cblas_sscal_k;
	//blas_kernels.dscal_k = cblas_dscal_k;
	//blas_kernels.saxpy_k = cblas_saxpy_k;
	//blas_kernels.daxpy_k = cblas_daxpy_k;
	//blas_kernels.saxpby_k = cblas_saxpby_k;
	//blas_kernels.daxpby_k = cblas_daxpby_k;

	// Initialize Level-2 kernel function pointers
	blas_kernels.sger_k = sger_k;
	blas_kernels.dger_k = dger_k;
	blas_kernels.sgemv_k = sgemv_k;
	blas_kernels.dgemv_k = dgemv_k;

	// Initialize Level-3 kernel function pointers
	blas_kernels.sgemm_k = sgemm_k;

	// Initialize kernel function pointers based on CPU features
	if (cpu_features & CPU_SSE)
	{
		blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc_sse;
		blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc_sse;
		blas_kernels.sasum_k_noinc = cblas_sasum_k_noinc_sse;
		blas_kernels.dasum_k_noinc = cblas_dasum_k_noinc_sse;
		blas_kernels.scopy_k_noinc = cblas_scopy_k_noinc_sse;
		blas_kernels.dcopy_k_noinc = cblas_dcopy_k_noinc_sse;

		if (cpu_features & CPU_AVX)
		{
			blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc_avx;
			blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc_avx;
			blas_kernels.sasum_k_noinc = cblas_sasum_k_noinc_avx;
			blas_kernels.dasum_k_noinc = cblas_dasum_k_noinc_avx;
			blas_kernels.scopy_k_noinc = cblas_scopy_k_noinc_avx;
			blas_kernels.dcopy_k_noinc = cblas_dcopy_k_noinc_avx;

			// Check for FMA3 support and dispatch accordingly
			if (cpu_features & CPU_x64_FMA3)
			{
				blas_kernels.sdot_k_noinc = cblas_sdot_k_noinc_fma;
				blas_kernels.ddot_k_noinc = cblas_ddot_k_noinc_fma;
				blas_kernels.sgemm_k = sgemm_k_fma;
			}
		}
	}
}

//------------------------------------------------------
// query for cpu features
//------------------------------------------------------
static unsigned int __cpu_get_features(void)
{
#if defined(_MSC_VER)
	int info[4];

	__cpuid(info, 1);
	if (info[ECX] & (1 << 20))
		cpu_features |= CPU_SSE;

	if (info[ECX] & (1 << 28))
		cpu_features |= CPU_AVX;

	if (info[ECX] & BIT(12))
		cpu_features |= CPU_x64_FMA3;

	__cpuid(info, 7);

	if (info[EBX] & (1 << 5))
		cpu_features |= CPU_AVX2;

	if (info[EBX] & (1 << 16))
		cpu_features |= CPU_AVX512;
	
	// Check for hybrid architecture (P-cores + E-cores)
	if (info[EDX] & (1 << 15))
		cpu_features |= CPU_HYBRID;
	
#else
	unsigned int eax, ebx, ecx, edx;

	__cpuid(1, eax, ebx, ecx, edx);
	if (ecx & (1 << 20))
		cpu_features |= CPU_SSE;

	if (ecx & (1 << 28))
		cpu_features |= CPU_AVX;

	if (ecx & BIT(12))
		cpu_features |= CPU_x64_FMA3;

	__cpuid_count(7, 0, eax, ebx, ecx, edx);

	if (ebx & BIT(5))
		cpu_features |= CPU_AVX2;

	if (ebx & BIT(16))
		cpu_features |= CPU_AVX512;

	// Check for hybrid architecture (P-cores + E-cores)
	if (edx & (1 << 15))
		cpu_features |= CPU_HYBRID;

#endif

	init_blas_kernels(cpu_features);

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
// return the number of usable cores
//------------------------------------------------------
int cpu_get_core_count(void)
{
	static int cores = -1;

	// use cached value if it exists
	if (-1 != cores)
		return cores;

#ifdef _WIN32
	SYSTEM_INFO si;
	GetSystemInfo(&si);

	cores = si.dwNumberOfProcessors;
	return cores;
#else
    cores = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (cores == -1)
        cores = 1;

    return cores;
#endif

#if defined(_MSC_VER)
	//int info[4];
	//const char* vendor_string = cpu_get_core_name();

	//if (!strcmp(vendor_string, "GenuineIntel"))
	//{
	//	__cpuid(info, 4);
	//	cores = ((info[EAX] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
	//}
	//else if (!strcmp(vendor_string, "AuthenticAMD"))
	//{
	//	__cpuid(info, 0x80000008);
	//	cores = ((unsigned)(info[ECX] & 0xff)) + 1; // ECX[7:0] + 1
	//}
	//else
	//{
	//	puts("Error: Unknown CPU vendor");
	//	cores = 1;
	//}

	//return cores;
#else 
	#ifdef __APPLE__
		uint32_t entry;
		size_t len = sizeof(entry);

		// TODO - per sysctl.h this might want to be hw.ncpu or physicalcpu_max?
		sysctlbyname("hw.physicalcpu", &entry, &len, NULL, 0);
		cores = (int)entry;

		return cores;
	#else
		unsigned int eax, ebx, ecx, edx;
		const char* vendor_string = cpu_get_core_name();

		if (!strcmp(vendor_string, "GenuineIntel"))
		{
			__cpuid(4, eax, ebx, ecx, edx);
			cores =  ((eax >> 26) & 0x3f) + 1; // EAX[31:26] + 1
		}
		else if (!strcmp(vendor_string, "AuthenticAMD"))
		{
			__cpuid(0x80000008, eax, ebx, ecx, edx);
			cores = ((unsigned)(ecx & 0xff)) + 1; // ECX[7:0] + 1
		}
		else
		{
			puts("Error: Unknown CPU vendor, defaulting to 1 core");
			cores = 1;
		}

		// printf("Detected %d cores\n", core_count);
		return cores;
	#endif
#endif

}

//------------------------------------------------------
// Check if CPU has hybrid architecture (P+E cores)
//------------------------------------------------------
int cpu_is_hybrid(void)
{
	static int hybrid_checked = 0;
	static int is_hybrid = 0;
	
	if (hybrid_checked)
		return is_hybrid;
	
	const char* vendor = cpu_get_core_name();
	
	// Only Intel has hybrid architecture so far
	if (strncmp(vendor, "GenuineIntel", 12) != 0)
	{
		hybrid_checked = 1;
		return 0;
	}
	
#if defined(_MSC_VER)
	int info[4];
	
	// Check CPUID leaf 0x07, subleaf 0, EDX bit 15 for hybrid support
	__cpuidex(info, 0x07, 0);
	if (info[EDX] & (1 << 15))
	{
		is_hybrid = 1;
	}
#else
	unsigned int eax, ebx, ecx, edx;
	
	__cpuid_count(0x07, 0, eax, ebx, ecx, edx);
	if (edx & (1 << 15))
	{
		is_hybrid = 1;
	}
#endif
	
	hybrid_checked = 1;
	return is_hybrid;
}

//------------------------------------------------------
// Get number of P-cores (performance cores)
//------------------------------------------------------
int cpu_get_p_core_count(void)
{
	if (!cpu_is_hybrid())
		return cpu_get_core_count();
	
	// For hybrid CPUs, we need to enumerate cores via Windows API
	// or use CPUID leaf 0x1A to identify core types
	// This is a simplified version - actual implementation would need
	// to iterate through all logical processors
	
#ifdef _WIN32
	// Windows provides this through GetLogicalProcessorInformationEx
	// but for now, we'll use a heuristic
	int total_cores = cpu_get_core_count();
	
	// Common Intel configurations:
	// 12th gen: 8P+8E (16 total)
	// 13th gen: 8P+16E (24 total), 6P+8E (14 total)
	// This is a rough estimate - real detection needs Windows API calls
	
	// For now, return approximately 1/2 to 2/3 as P-cores
	return (total_cores * 2) / 3;
#else
	return cpu_get_core_count() / 2;
#endif
}

//------------------------------------------------------
// Get number of E-cores (efficiency cores)
//------------------------------------------------------
int cpu_get_e_core_count(void)
{
	if (!cpu_is_hybrid())
		return 0;
	
	return cpu_get_core_count() - cpu_get_p_core_count();
}

