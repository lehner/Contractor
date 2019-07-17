SRCS = $(wildcard *.cc)

PROGS = $(patsubst main%.cc,exe%,$(SRCS))

all: $(PROGS)

exe: main.cc
	mpiicpc -O3 -mkl -xmic-avx512 -fopenmp -std=c++11 -o $@ -lz $<

#
#-mcmodel=large
