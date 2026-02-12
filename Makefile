# get arch name
ARCH = $(shell uname -m)
TARGET = blas_test
OBJS = swap.o dot.o copy.o axpy.o scal.o axpby.o asum.o nrm2.o rot.o ger.o \
	gemv.o gemm.o rotg.o util.o server.o setv.o kernels/dot_k.o \
	kernels/dot_k_neon.o kernels/dot_k_sse.o kernels/dot_k_avx.o kernels/dot_k_fma.o kernels/dot_k_avx512.o \
	kernels/asum_k_sse.o kernels/asum_k_avx.o kernels/asum_k_neon.o \
	kernels/copy_k_sse.o kernels/copy_k_avx.o kernels/copy_k_neon.o \
	kernels/swap_k_sse.o kernels/swap_k_avx.o kernels/swap_k_neon.o \
	kernels/setv_k_sse.o kernels/setv_k_avx.o kernels/setv_k_neon.o \
	kernels/rot_k_sse.o kernels/rot_k_avx.o kernels/rot_k_neon.o \
	kernels/nrm2_k.o kernels/nrm2_k_sse.o kernels/nrm2_k_avx.o kernels/nrm2_k_fma.o kernels/nrm2_k_neon.o \
	kernels/scal_k.o kernels/scal_k_sse.o kernels/scal_k_avx.o kernels/scal_k_avx512.o kernels/scal_k_neon.o \
	kernels/axpy_k_sse.o kernels/axpy_k_avx.o kernels/axpy_k_fma.o kernels/axpy_k_avx512.o kernels/axpy_k_neon.o \
	kernels/axpby_k_sse.o kernels/axpby_k_avx.o kernels/axpby_k_fma.o kernels/axpby_k_neon.o \
	kernels/ger_k.o kernels/ger_k_neon.o kernels/ger_k_fma.o \
	kernels/gemv_k.o kernels/gemv_k_avx.o kernels/gemv_k_fma.o kernels/gemv_k_neon.o \
	kernels/gemm_k.o kernels/gemm_k_sse.o kernels/gemm_k_avx.o kernels/gemm_k_fma.o kernels/gemm_k_neon.o \
	kernels/dgemm_k_sse.o kernels/dgemm_k_avx.o kernels/dgemm_k_fma.o kernels/dgemm_k_neon.o
DEPS = cblas.h cblas_config.h tests/test.h platform/threading.h platform/simd.h platform/cpuid.h
CFLAGS += -g -O2 -Wall -Wextra -Wpedantic -I. -Itests #-DNDEBUG
LIBNAME = libcblas.a
LFLAGS += -L. -lcblas -lm

# Configuration options - can be overridden on command line
# e.g., make CBLAS_ENABLE_MT=0 to disable multi-threading
CBLAS_ENABLE_MT ?= 1
CBLAS_CHECK_INPUTS ?= 1
CBLAS_USE_STATIC_BUFFERS ?= 1
CBLAS_MAX_THREADS ?= 64

# add Intel specific compiler flags - per-file basis for SIMD kernels
# NOTE: We no longer set global AVX2 flags to support older CPUs
# SSE kernels get -msse4.1, AVX kernels get -mavx, FMA kernels get -mavx2 -mfma
ifeq ($(ARCH), x86_64)
	# Per-file flags are set below using pattern rules
endif

# add ARM64 cpuid code
ifeq ($(ARCH), arm64)
	OBJS += cpuid_arm64.o
else
	ifeq ($(ARCH), aarch64)
		OBJS += cpuid_arm64.o
	else
		OBJS += cpuid_x64.o
	endif
endif

all: cblas_config.h $(LIBNAME) axpy_perf blas_stress blas_test test_strided test_stats test_threshold test_dot_threshold test_concurrent test_mt_debug test_level2_mt test_autotune test_gemm_accuracy gemm_perf dgemm_perf ger_perf dger_perf gemv_perf dot_perf dot_threshold_tuning nrm2_asum_rot_perf copy_perf

# Generate cblas_config.h from configuration variables
cblas_config.h: cblas_config.h.in Makefile
	@echo "Generating cblas_config.h..."
	@sed -e 's/@CBLAS_MAX_THREADS@/$(CBLAS_MAX_THREADS)/g' \
	     -e 's/#cmakedefine01 CBLAS_ENABLE_MT/#define CBLAS_ENABLE_MT $(CBLAS_ENABLE_MT)/g' \
	     -e 's/#cmakedefine01 CBLAS_CHECK_INPUTS/#define CBLAS_CHECK_INPUTS $(CBLAS_CHECK_INPUTS)/g' \
	     -e 's/#cmakedefine01 CBLAS_USE_STATIC_BUFFERS/#define CBLAS_USE_STATIC_BUFFERS $(CBLAS_USE_STATIC_BUFFERS)/g' \
	     cblas_config.h.in > cblas_config.h
	
$(LIBNAME): $(OBJS)
	ar rcs $(LIBNAME) $(OBJS)

# Tests
blas_stress: $(LIBNAME) tests/test_main.o tests/test_stress.o
	$(CC) -o $@ $^ $(LFLAGS)

blas_test: $(LIBNAME) tests/test_main.o tests/test.o
	$(CC) -o $@ $^ $(LFLAGS)

test_strided: $(LIBNAME) tests/test_main.o tests/test_strided.o
	$(CC) -o $@ $^ $(LFLAGS)

test_stats: $(LIBNAME) tests/test_stats.o
	$(CC) -o $@ $^ $(LFLAGS)

test_overhead: $(LIBNAME) tests/test_overhead.o
	$(CC) -o $@ $^ $(LFLAGS)

test_threshold: $(LIBNAME) tests/test_threshold.o
	$(CC) -o $@ $^ $(LFLAGS)

test_dot_threshold: $(LIBNAME) tests/test_dot_threshold.o
	$(CC) -o $@ $^ $(LFLAGS)

test_concurrent: $(LIBNAME) tests/test_main.o tests/test_concurrent.o
	$(CC) -o $@ $^ $(LFLAGS) -lpthread

test_mt_debug: $(LIBNAME) tests/test_mt_debug.o
	$(CC) -o $@ $^ $(LFLAGS)

test_level2_mt: $(LIBNAME) tests/test_main.o tests/test_level2_mt.o
	$(CC) -o $@ $^ $(LFLAGS)

test_autotune: $(LIBNAME) tests/test_main.o tests/test_autotune.o
	$(CC) -o $@ $^ $(LFLAGS)

test_gemm_accuracy: $(LIBNAME) tests/test_main.o tests/test_gemm_accuracy.o
	$(CC) -o $@ $^ $(LFLAGS)

# Benchmarks
gemm_perf: $(LIBNAME) benchmarks/gemm_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

dgemm_perf: $(LIBNAME) benchmarks/dgemm_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

ger_perf: $(LIBNAME) benchmarks/ger_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

dger_perf: $(LIBNAME) benchmarks/dger_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

dot_perf: $(LIBNAME) benchmarks/dot_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

axpy_perf: $(LIBNAME) benchmarks/axpy_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

copy_perf: $(LIBNAME) benchmarks/copy_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

dot_threshold_tuning: $(LIBNAME) benchmarks/dot_threshold_tuning.o
	$(CC) -o $@ $^ $(LFLAGS)

dot_threshold_tuning_large: $(LIBNAME) benchmarks/dot_threshold_tuning_large.o
	$(CC) -o $@ $^ $(LFLAGS)

nrm2_asum_rot_perf: $(LIBNAME) benchmarks/nrm2_asum_rot_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

gemv_perf: $(LIBNAME) benchmarks/gemv_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

mem_perf: $(LIBNAME) benchmarks/mem_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

# Per-file SIMD compile flags for x86-64 kernels
# Only apply on x86-64, not on ARM64
ifeq ($(ARCH), x86_64)
# SSE kernels - 128-bit, works on Sandy Bridge and later
kernels/%_sse.o: kernels/%_sse.c $(DEPS)
	$(CC) -c $(CFLAGS) -msse4.1 $(CPPFLAGS) $< -o $@

# AVX kernels - 256-bit, requires AVX2 (Haswell and later)
# Note: 256-bit __m256 requires AVX2, Sandy Bridge only has 128-bit AVX
kernels/%_avx.o: kernels/%_avx.c $(DEPS)
	$(CC) -c $(CFLAGS) -mavx2 $(CPPFLAGS) $< -o $@

# FMA kernels - 256-bit + FMA (Haswell and later)
kernels/%_fma.o: kernels/%_fma.c $(DEPS)
	$(CC) -c $(CFLAGS) -mavx2 -mfma $(CPPFLAGS) $< -o $@

# AVX512 kernels - Skylake-X and later
kernels/%_avx512.o: kernels/%_avx512.c $(DEPS)
	$(CC) -c $(CFLAGS) -mavx512f $(CPPFLAGS) $< -o $@

# Mixed-SIMD kernels (contain both SSE and AVX code, need AVX2 for 256-bit)
kernels/ger_k.o: kernels/ger_k.c $(DEPS)
	$(CC) -c $(CFLAGS) -mavx2 $(CPPFLAGS) $< -o $@
endif

# Pattern rules for subdirectories
tests/%.o: tests/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

benchmarks/%.o: benchmarks/%.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

install:
	sudo mkdir -p /opt/cblas/lib /opt/cblas/include/platform
	sudo cp libcblas.a /opt/cblas/lib
	sudo cp cblas.h /opt/cblas/include
	sudo cp cblas_config.h /opt/cblas/include
	sudo cp platform/cpuid.h /opt/cblas/include/platform

test: all
	./blas_test

clean:
	rm -f $(TARGET) $(OBJS) $(LIBNAME) cblas_config.h
	rm -f tests/*.o benchmarks/*.o
	rm -f blas_test blas_stress test_strided test_stats test_overhead test_threshold
	rm -f test_dot_threshold test_concurrent test_mt_debug test_level2_mt test_autotune test_gemm_accuracy
	rm -f gemm_perf dgemm_perf ger_perf dger_perf dot_perf axpy_perf copy_perf gemv_perf mem_perf
	rm -f dot_threshold_tuning dot_threshold_tuning_large nrm2_asum_rot_perf
