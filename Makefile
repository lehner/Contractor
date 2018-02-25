SRCS = $(wildcard *.cc)

PROGS = $(patsubst main-%.cc,exe-%,$(SRCS))

all: $(PROGS)

exe-%: main-%.cc
	icpc -O3 -mcmodel=large -qopenmp -std=c++11 -o $@ -lz $<

