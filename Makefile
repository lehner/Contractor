SRCS = $(wildcard *.cc)

PROGS = $(patsubst main%.cc,exe-test%,$(SRCS))

all: $(PROGS)

exe-test: main.cc
	mpiicpc -O3 -mkl -xmic-avx512 -fopenmp -std=c++11 -o $@ -lz $<

#
#-mcmodel=large
