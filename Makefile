# get arch name
ARCH = $(shell uname -m)
TARGET = blas_test
OBJS = swap.o dot.o copy.o axpy.o scal.o axpby.o asum.o nrm2.o rot.o ger.o gemv.o rotg.o util.o server.o test_main.o
DEPS = cblas.h test.h
CFLAGS += -g -O3
LIBNAME = libcblas.a

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

all: $(TARGET) blas_stress
	
lib: $(OBJS)
	ar rcs $(LIBNAME) $(OBJS)

blas_stress: $(OBJS) test_stress.o
	$(CC) $(LFLAGS) -o $@ $^

$(TARGET):	$(OBJS) test.o
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS) test.o test_stress.o blas_stress

