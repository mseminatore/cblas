// Copyright 2022 Mark Seminatore. All rights reserved.
#pragma once

#ifndef __TEST_H
#define __TEST_H

#include "assert.h"

#ifdef _WIN32
#   define CHECK_MARK "\xFB"
#else
#   define CHECK_MARK "\u2713"
#endif

// simple test harness
#define BEGIN_TESTS()   puts("Beginning test cases...")
#define TEST(s)         printf("\t%d checking that: " #s " ", ++test_number); assert(s); printf(CHECK_MARK); putchar('\n')
#define SUITE(s)        puts("\nTesting suite " s "...\n"); test_suites++
#define END_TESTS()     printf("\n...end test cases.\nSuccessfully completed %d tests in %d suites.\n\n", test_number, test_suites)

#ifndef TRUE
#	define TRUE 1
#endif

#ifndef FALSE
#	define FALSE 0
#endif

#define ARRAY_SIZE(a)           (sizeof(a)/sizeof(a[0]))

#define EQUAL_ARRAY(a, b)       !memcmp((a), (b), sizeof(a))
#define NOT_EQUAL_ARRAY(a, b)   !EQUAL_ARRAY(a, b)

// void test_string();
// void test_stdlib();
// void test_ctype();
// void test_stdio();

extern int test_number;

#endif // #ifndef __TEST_H
