# get arch name
ARCH = $(shell uname -m)
TARGET = blas_test
OBJS = swap.o dot.o copy.o axpy.o scal.o axpby.o asum.o nrm2.o rot.o ger.o \
	gemv.o gemm.o rotg.o util.o server.o test_main.o
DEPS = cblas.h test.h
CFLAGS += -g -O2 #-DNDEBUG
LIBNAME = libcblas.a
#LFLAGS += -lcblas

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

# add ARM64 cpuid code
ifeq ($(ARCH), arm64)
	OBJS += cpuid_arm64.o
else
	OBJS += cpuid_x64.o
endif

all: $(LIBNAME) blas_stress blas_test gemm_perf ger_perf
	
$(LIBNAME): $(OBJS)
	ar rcs $(LIBNAME) $(OBJS)

blas_stress: $(LIBNAME) test_stress.o
	$(CC) -o $@ $^ $(LFLAGS)

blas_test: $(LIBNAME) test.o
	$(CC) -o $@ $^ $(LFLAGS)

gemm_perf: $(LIBNAME) gemm_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

ger_perf: $(LIBNAME) ger_perf.o
	$(CC) -o $@ $^ $(LFLAGS)

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

install:
	sudo mkdir -p /opt/cblas/lib /opt/cblas/include
	sudo cp libcblas.a /opt/cblas/lib
	sudo cp cblas.h /opt/cblas/include

clean:
	rm $(TARGET) $(OBJS) test.o test_stress.o blas_perf.o blas_stress blas_test gemm_perf ger_perf

