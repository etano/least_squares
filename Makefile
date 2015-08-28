.PHONY: all clean cmpfit

.DEFAULT: all

CXX = g++

CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(CMPFITFLAGS)

TARGETS = least_squares

all: cmpfit $(TARGETS)

cmpfit:
	$(MAKE) -C cmpfit

clean:
	rm -f $(TARGETS) main.o

main.o : main.cpp *.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGETS) : main.o cmpfit/mpfit.o
	$(CXX) $(CXXFLAGS) -o $@ main.o cmpfit/mpfit.o
