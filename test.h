//--------------------------------------------------------
// Simple C/C++ test framework
// 
// Copyright 2022 Mark Seminatore. All rights reserved.
//--------------------------------------------------------
#pragma once

#ifndef __TEST_H
#define __TEST_H

#include "assert.h"

// screen control
#define ESC "\x1b"
#define TERM_CLEAR ESC "[2J" ESC "[H"

// text colors
#define TERM_RESET ESC "[0m"
#define TERM_BLACK ESC "[30m"
#define TERM_RED ESC "[31m"
#define TERM_GREEN ESC "[32m"
#define TERM_YELLOW ESC "[33m"
#define TERM_BLUE ESC "[34m"
#define TERM_MAGENTA ESC "[35m"
#define TERM_CYAN ESC "[36m"
#define TERM_WHITE ESC "[37m"

#ifdef _WIN32
#   define CHECK_MARK TERM_GREEN "\xFB" TERM_RESET
#   define X_MARK TERM_RED "X" TERM_RESET
#else
#   define CHECK_MARK TERM_GREEN "\u2713" TERM_RESET
#   define X_MARK TERM_RED "\u274C" TERM_RESET
#endif

#if 1
#   define TEST_ASSERT(expr) if ((expr)) { puts(CHECK_MARK);} else { puts(X_MARK); test_failures++;} 
#else
#   define TEST_ASSERT(expr) assert(expr); printf(CHECK_MARK); putchar('\n')
#endif

// simple test harness
#define BEGIN_TESTS()   puts("Beginning test cases...")
#define TEST(s)         printf("\t%d checking that: " #s " ", ++test_number); TEST_ASSERT(s)
#define SUITE(s)        puts("\nTesting suite " s "...\n"); test_suites++
#define END_TESTS()     printf("\n...finished test cases.\nSuccessfully evaluated %s%d%s tests in %s%d%s suites, with %s%d%s failed test cases.\n\n", TERM_GREEN, test_number, TERM_RESET, TERM_GREEN, test_suites, TERM_RESET, test_failures ? TERM_RED : TERM_GREEN, test_failures, TERM_RESET); return test_failures

#ifndef TRUE
#	define TRUE 1
#endif

#ifndef FALSE
#	define FALSE 0
#endif

#define ARRAY_SIZE(a)           (sizeof(a)/sizeof(a[0]))

#define EQUAL_ARRAY(a, b)       !memcmp((a), (b), sizeof(a))
#define NOT_EQUAL_ARRAY(a, b)   !EQUAL_ARRAY(a, b)

extern int test_number;
extern int test_suites;
extern int test_failures;

#endif // #ifndef __TEST_H
