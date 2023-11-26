//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>

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
// return the number of usable cores
//------------------------------------------------------
int cpu_get_core_count()
{
	const char* vendor_string = cpu_get_core_name();

#if defined(_MSC_VER)
	int info[4];

	if (!strcmp(vendor_string, "GenuineIntel"))
	{
		__cpuid(info, 4);
		return ((info[EAX] >> 26) & 0x3f) + 1; // EAX[31:26] + 1
	}
	else if (!strcmp(vendor_string, "AuthenticAMD"))
	{
		__cpuid(info, 0x80000008);
		return ((unsigned)(info[ECX] & 0xff)) + 1; // ECX[7:0] + 1
	}
	else
	{
		puts("Error: Unknown CPU vendor");
		return 1;
	}
#else 
	#ifdef __APPLE__
		uint32_t entry;
		size_t len = sizeof(entry);

		// TODO - per sysctl.h this might want to be hw.ncpu or physicalcpu_max?
		sysctlbyname("hw.physicalcpu", &entry, &len, NULL, 0);
		return entry;
	#else
		unsigned int eax, ebx, ecx, edx;
		int core_count = 1;

		if (!strcmp(vendor_string, "GenuineIntel"))
		{
			__cpuid(4, eax, ebx, ecx, edx);
			core_count =  ((eax >> 26) & 0x3f) + 1; // EAX[31:26] + 1
		}
		else if (!strcmp(vendor_string, "AuthenticAMD"))
		{
			__cpuid(0x80000008, eax, ebx, ecx, edx);
			core_count = ((unsigned)(ecx & 0xff)) + 1; // ECX[7:0] + 1
		}
		else
		{
			puts("Error: Unknown CPU vendor, defaulting to 1 core");
		}

		printf("Detected %d cores\n", core_count);
		return core_count;
	#endif
#endif

}