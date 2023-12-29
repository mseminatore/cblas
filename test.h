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
#define TERM_BOLD ESC "[1m"
#define TERM_UNDERLINE ESC "[4m"
#define TERM_BLINK ESC "[5m"

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
#define TERM_BRIGHT_BLACK ESC "[90m"
#define TERM_BRIGHT_RED ESC "[91m"
#define TERM_BRIGHT_GREEN ESC "[92m"
#define TERM_BRIGHT_YELLOW ESC "[93m"
#define TERM_BRIGHT_BLUE ESC "[94m"
#define TERM_BRIGHT_MAGENTA ESC "[95m"
#define TERM_BRIGHT_CYAN ESC "[96m"
#define TERM_BRIGHT_WHITE ESC "[97m"

#ifdef _WIN32
#   define CHECK_MARK TERM_GREEN "\xFB" TERM_RESET
#   define X_MARK TERM_BRIGHT_RED "X" TERM_RESET
#else
#   define CHECK_MARK TERM_GREEN "\u2713" TERM_RESET
#   define X_MARK TERM_BRIGHT_RED "\u274C" TERM_RESET
#endif

#if 1
#   define TEST_ASSERT(expr, f, l) if ((expr)) { puts(CHECK_MARK);} else { puts(X_MARK); test_failures++; printf("\tTest %sfailed%s at [%s%s:%d%s]\n", TERM_BRIGHT_RED, TERM_RESET, TERM_BRIGHT_RED, (f), (l), TERM_RESET); } 
#else
#   define TEST_ASSERT(expr) assert(expr); printf(CHECK_MARK); putchar('\n')
#endif

// simple test harness
#define BEGIN_TESTS()   printf("%sBegin test pass%s...\n", TERM_BRIGHT_MAGENTA, TERM_RESET)
#define TEST(s)         printf("\t%d test case: %s" #s "%s ", ++test_number, TERM_CYAN, TERM_RESET); TEST_ASSERT(s, __FILE__, __LINE__)
#define TESTEX(msg,s)   printf("\t%d test case: %s" msg "%s ", ++test_number, TERM_CYAN, TERM_RESET); TEST_ASSERT(s, __FILE__, __LINE__)
#define COMMENT(s)      printf("\t%s" s "%s\n", TERM_BRIGHT_BLUE, TERM_RESET)
#define SUITE(s)        printf("\nTesting suite %s" s "%s...\n", TERM_YELLOW, TERM_RESET); test_suites++
#define MODULE(s)       printf("\nModule %s" s "%s...\n", TERM_BRIGHT_MAGENTA, TERM_RESET); test_modules++
#define END_TESTS()     printf("\n%sTest pass completed%s.\nEvaluated %s%d%s modules, %s%d%s suites, and %s%d%s tests with %s%d%s failed test case(s).\n\n", TERM_BRIGHT_MAGENTA, TERM_RESET, TERM_GREEN, test_modules, TERM_RESET, TERM_GREEN, test_suites, TERM_RESET, TERM_GREEN, test_number, TERM_RESET, test_failures ? TERM_BRIGHT_RED : TERM_GREEN, test_failures, TERM_RESET); return test_failures

#ifndef TRUE
#	define TRUE 1
#endif

#ifndef FALSE
#	define FALSE 0
#endif

#ifndef EPSILON
#   define EPSILON 1e-5
#endif

#define EQUAL_EPSILON(a, b)     (fabs((a) - (b)) < EPSILON)

#ifndef ARRAY_SIZE
#	define ARRAY_SIZE(a)        (sizeof(a)/sizeof(a[0]))
#endif

#define EQUAL_ARRAY(a, b)       !memcmp((a), (b), sizeof(a))
#define NOT_EQUAL_ARRAY(a, b)   !EQUAL_ARRAY(a, b)

extern int test_number;
extern int test_suites;
extern int test_failures;
extern int test_modules;

int test_main(int argc, char* argv[]);

#endif // #ifndef __TEST_H
