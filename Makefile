# get arch name
ARCH = $(shell uname -m)
TARGET = blas_test
OBJS = swap.o dot.o copy.o axpy.o scal.o axpby.o asum.o nrm2.o rot.o ger.o \
	gemv.o gemm.o rotg.o util.o server.o setv.o
DEPS = cblas.h cblas_config.h test.h
CFLAGS += -g -O2 -Wall -Wextra -Wpedantic #-DNDEBUG
LIBNAME = libcblas.a
LFLAGS += -L. -lcblas -lm

# Configuration options - can be overridden on command line
# e.g., make CBLAS_ENABLE_MT=0 to disable multi-threading
CBLAS_ENABLE_MT ?= 1
CBLAS_USE_SIMD ?= 1
CBLAS_CHECK_INPUTS ?= 1
CBLAS_USE_STATIC_BUFFERS ?= 1
CBLAS_MAX_THREADS ?= 64

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
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

all: cblas_config.h $(LIBNAME) blas_stress blas_test test_strided test_stats test_threshold test_concurrent test_mt_debug gemm_perf ger_perf dot_perf nrm2_asum_rot_perf

# Generate cblas_config.h from configuration variables
cblas_config.h: cblas_config.h.in Makefile
	@echo "Generating cblas_config.h..."
	@sed -e 's/@CBLAS_MAX_THREADS@/$(CBLAS_MAX_THREADS)/g' \
	     -e 's/#cmakedefine01 CBLAS_ENABLE_MT/#define CBLAS_ENABLE_MT $(CBLAS_ENABLE_MT)/g' \
	     -e 's/#cmakedefine01 CBLAS_USE_SIMD/#define CBLAS_USE_SIMD $(CBLAS_USE_SIMD)/g' \
	     -e 's/#cmakedefine01 CBLAS_CHECK_INPUTS/#define CBLAS_CHECK_INPUTS $(CBLAS_CHECK_INPUTS)/g' \
	     -e 's/#cmakedefine01 CBLAS_USE_STATIC_BUFFERS/#define CBLAS_USE_STATIC_BUFFERS $(CBLAS_USE_STATIC_BUFFERS)/g' \
	     cblas_config.h.in > cblas_config.h
	
$(LIBNAME): $(OBJS)
	ar rcs $(LIBNAME) $(OBJS)

blas_stress: $(LIBNAME) test_main.o test_stress.o
	$(CC) -o $@ $^ $(LFLAGS)

blas_test: $(LIBNAME) test_main.o test.o
	$(CC) -o $@ $^ $(LFLAGS)

test_strided: $(LIBNAME) test_main.o test_strided.o
	$(CC) -o $@ $^ $(LFLAGS)

test_stats: $(LIBNAME) test_stats.o
	$(CC) -o $@ $^ $(LFLAGS)

test_overhead: $(LIBNAME) test_overhead.o
	$(CC) -o $@ $^ $(LFLAGS)

test_threshold: $(LIBNAME) test_threshold.o
	$(CC) -o $@ $^ $(LFLAGS)

test_concurrent: $(LIBNAME) test_main.o test_concurrent.o
	$(CC) -o $@ $^ $(LFLAGS) -lpthread

test_mt_debug: $(LIBNAME) test_mt_debug.o
	$(CC) -o $@ $^ $(LFLAGS)

gemm_perf: $(LIBNAME) gemm_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

ger_perf: $(LIBNAME) ger_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

dot_perf: $(LIBNAME) dot_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

nrm2_asum_rot_perf: $(LIBNAME) nrm2_asum_rot_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

install:
	sudo mkdir -p /opt/cblas/lib /opt/cblas/include
	sudo cp libcblas.a /opt/cblas/lib
	sudo cp cblas.h /opt/cblas/include
	sudo cp cblas_config.h /opt/cblas/include

test: all
	./blas_test

clean:
	rm -f $(TARGET) $(OBJS) $(LIBNAME) test_main.o test.o test_stress.o blas_stress blas_test.o blas_test test_threshold.o test_threshold test_concurrent.o test_concurrent gemm_perf.o gemm_perf ger_perf.o ger_perf dot_perf.o dot_perf nrm2_asum_rot_perf.o nrm2_asum_rot_perf cblas_config.h test_strided.o test_strided test_stats.o test_stats

