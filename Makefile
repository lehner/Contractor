SRCS = $(wildcard *.cc)

PROGS = $(patsubst main%.cc,exe%,$(SRCS))

all: $(PROGS)

exe: main.cc
	mpic++ -O3 -fopenmp -std=c++11 -o $@ -lz -lcblas $<

#
#-mcmodel=large
