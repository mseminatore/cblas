//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include "cblas.h"

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

//------------------------------------------------------
// return the CPU name
//------------------------------------------------------
const char *cpu_get_core_name()
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
int cpu_get_cacheline_size()
{
	int line_size = 64;

#ifdef __APPLE__
	uint32_t entry;
	size_t len = sizeof(entry);

	sysctlbyname("hw.cachelinesize", &entry, &len, NULL, 0);
	return entry;
#else

#if defined(_MSC_VER)
	uint32_t regs[4];
	__cpuid(regs, 0x80000006);
	unsigned lsize = regs[ECX] & 0xff;
#endif

#endif

	return line_size;
}

//------------------------------------------------------
// return the L2$ size
//------------------------------------------------------
int cpu_get_l2_cache_size()
{
	uint32_t l2_cache_size = 0;

#if defined(_MSC_VER)
	uint32_t regs[4];
	__cpuid(regs, 0x80000006);
	l2_cache_size = (regs[ECX] >> 16) & 0xFFFF; // Extract L2 cache size in KB
#endif

	return l2_cache_size;
}

//------------------------------------------------------
// return the CPU core brand name
//------------------------------------------------------
const char* cpu_get_brand_string(void)
{
#if defined(_MSC_VER)
	unsigned int regs[12];
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
//
//------------------------------------------------------
unsigned int cpu_get_features()
{
	unsigned int features = CPU_NONE;

#if defined(_MSC_VER)
	int info[4];

	__cpuid(info, 1);
	if (info[ECX] & (1 << 20))
		features |= CPU_SSE;

	if (info[ECX] & (1 << 28))
		features |= CPU_AVX;

	__cpuid(info, 7);

	if (info[EBX] & (1 << 5))
		features |= CPU_AVX2;

	if (info[EBX] & (1 << 16))
		features |= CPU_AVX512;
	
#else
	unsigned int eax, ebx, ecx, edx;

	__cpuid(1, eax, ebx, ecx, edx);
	if (ecx & (1 << 20))
		features |= CPU_SSE;

	if (ecx & (1 << 28))
		features |= CPU_AVX;

	__cpuid(7, eax, ebx, ecx, edx);

	if (ebx & (1 << 5))
		features |= CPU_AVX2;

	if (ebx & (1 << 16))
		features |= CPU_AVX512;

#endif

	return features;
}

//------------------------------------------------------
// return the number of usable cores
//------------------------------------------------------
int cpu_get_core_count()
{
	const char* vendor_string = cpu_get_core_name();
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
	int info[4];

	if (!strcmp(vendor_string, "GenuineIntel"))
	{
		__cpuid(info, 4);
		cores = ((info[EAX] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
	}
	else if (!strcmp(vendor_string, "AuthenticAMD"))
	{
		__cpuid(info, 0x80000008);
		cores = ((unsigned)(info[ECX] & 0xff)) + 1; // ECX[7:0] + 1
	}
	else
	{
		puts("Error: Unknown CPU vendor");
		cores = 1;
	}

	return cores;
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