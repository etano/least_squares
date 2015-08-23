.PHONY: all clean

.DEFAULT: all

CXX = g++

NLOPTFLAGS = -I${HOME}/include -L${HOME}/lib -lnlopt -lm
CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(NLOPTFLAGS)

TARGETS = least_squares

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : least_squares.cpp *.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
