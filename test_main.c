//------------------------------------------------------
//
// Copyright 2023 Mark Seminatore. All rights reserved.
//------------------------------------------------------
#include <stdio.h>
#include "test.h"

#if defined(_WIN32) || defined(_WIN64)
#	include <Windows.h>
#endif

//
int test_number		= 0;
int test_failures 	= 0;
int test_suites 	= 0;
int test_modules 	= 0;

//------------------------------------------------------
// enable VT100 support in pre Win11 console window
//------------------------------------------------------
int setupConsole()
{
#if defined(_WIN32) || defined(_WIN64)
	// Set output mode to handle virtual terminal sequences
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	if (hOut == INVALID_HANDLE_VALUE)
	{
		return GetLastError();
	}

	DWORD dwMode = 0;
	if (!GetConsoleMode(hOut, &dwMode))
	{
		return GetLastError();
	}

	dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
	if (!SetConsoleMode(hOut, dwMode))
	{
		return GetLastError();
	}
#endif

	return 0;
}

//------------------------------------------------------
// test main entry point
//------------------------------------------------------
int main(int argc, char *argv[])
{
	setupConsole();

	BEGIN_TESTS();

    test_main(argc, argv);

	END_TESTS();
}