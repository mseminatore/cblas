//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include <string.h>

#define EAX 0
#define EBX 1
#define ECX 2
#define EDX 3

//------------------------------------------------------
//
//------------------------------------------------------
const char *cpu_get_core_name()
{
#if defined(_MSC_VER)
	static char mfgID[13];
	int info[4];

	__cpuid(info, 0);

	((int*)mfgID)[0] = info[1];
	((int*)mfgID)[1] = info[3];
	((int*)mfgID)[2] = info[2];
	mfgID[12] = 0;

	return mfgID;
#else
	return "Generic x64";
#endif
}

//------------------------------------------------------
//
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
int cpu_get_core_count()
{
#if defined(_MSC_VER)
	int info[4];

	const char* vendor_string = cpu_get_core_name();

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
#else
	{
		puts("Error: Unknown CPU vendor");
		return 1;
	}
#endif
}