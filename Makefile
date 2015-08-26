.PHONY: all clean

.DEFAULT: all

CXX = g++

NLOPTFLAGS = -I${HOME}/include -L${HOME}/lib -lnlopt -lm
CMPFITFLAGS = -L./cmpfit -lmpfit -lm
CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(NLOPTFLAGS) $(CMPFITFLAGS)

TARGETS = least_squares

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : main.cpp *.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
