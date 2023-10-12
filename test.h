// Copyright 2022 Mark Seminatore. All rights reserved.
#pragma once

#ifndef __TEST_H
#define __TEST_H

#include "assert.h"

// simple test harness
#define TEST(s)     printf("\t%d checking that: " #s " ", ++testnum); assert(s); putchar(251); putchar('\n')
#define SUITE(s)    puts("\nTesting " s "...\n")

#ifndef TRUE
#	define TRUE 1
#endif

#ifndef FALSE
#	define FALSE 0
#endif

void test_string();
void test_stdlib();
void test_ctype();
void test_stdio();

extern int testnum;

#endif // #ifndef __TEST_H
