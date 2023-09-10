# get arch name
ARCH = $(shell uname -m)

# add Intel specific compiler flags
ifeq ($(ARCH), x86_64)
	CFLAGS += -mavx2 -mfma
endif

TARGET = cblas
OBJS = swap.o test.o
DEPS = cblas.h
CFLAGS += -g -O3

all: cblas

$(TARGET):	$(OBJS) mnist.o
	$(CC) $(LFLAGS) -o $@ $^

test: $(OBJS) test.o
	$(CC) $(LFLAGS) -o $@ $^

%.o: %.c $(DEPS)
	$(CC) -c $(CFLAGS) $(CPPFLAGS) $< -o $@

clean:
	rm $(TARGET) $(OBJS) test

