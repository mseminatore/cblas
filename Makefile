# get arch name
ARCH = $(shell uname -m)
TARGET = blas_test
OBJS = swap.o dot.o copy.o axpy.o scal.o axpby.o asum.o nrm2.o rot.o ger.o \
	gemv.o gemm.o rotg.o util.o server.o setv.o
DEPS = cblas.h test.h
CFLAGS += -g -O2 -Wall -Wextra -Wpedantic #-DNDEBUG
LIBNAME = libcblas.a
LFLAGS += -L. -lcblas -lm

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

all: $(LIBNAME) blas_stress blas_test test_strided test_stats gemm_perf ger_perf dot_perf nrm2_asum_rot_perf
	
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

test: all
	./blas_test

clean:
	rm $(TARGET) $(OBJS) $(LIBNAME) test_main.o test.o test_stress.o blas_stress blas_test.o blas_test gemm_perf.o gemm_perf ger_perf.o ger_perf dot_perf.o dot_perf nrm2_asum_rot_perf.o nrm2_asum_rot_perf

