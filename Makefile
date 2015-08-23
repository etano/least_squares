.PHONY: all clean least_squares

.DEFAULT: all

CXX = g++

NLOPTFLAGS = -I${HOME}/include -L${HOME}/lib -lnlopt -lm
CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(NLOPTFLAGS)

TARGETS = least_squares

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

least_squares : least_squares.cpp least_squares.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
