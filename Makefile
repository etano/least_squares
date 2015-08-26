.PHONY: all clean cmpfit

.DEFAULT: all

CXX = g++

CMPFITFLAGS = -L./cmpfit -lmpfit -lm
CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(NLOPTFLAGS) $(CMPFITFLAGS)

TARGETS = least_squares

all: cmpfit $(TARGETS)

cmpfit:
	$(MAKE) -C cmpfit

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : main.cpp *.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<
