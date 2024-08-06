SRCS = $(wildcard *.cc)

PROGS = $(patsubst main%.cc,exe%,$(SRCS))

all: $(PROGS)

exe: main.cc
	mpicxx -O3 -fopenmp -std=c++11 -o $@ -lz -lcblas $<

#
#-mcmodel=large
