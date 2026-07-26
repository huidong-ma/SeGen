CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
LDLIBS ?= -lcrypto

.PHONY: all clean

all: prsec

prsec: prsec.cpp bloom_filter.hpp
	$(CXX) $(CXXFLAGS) prsec.cpp $(LDLIBS) -o $@

clean:
	rm -f prsec
