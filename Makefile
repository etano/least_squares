.PHONY: all clean cmpfit

.DEFAULT: all

CXX = clang++

NLOPTFLAGS = -I${HOME}/include
NLOPTLIBS = -L${HOME}/lib -lnlopt -lm
CMPFITLIBS = cmpfit/mpfit.o
CERESFLAGS = -I/usr/local/include -I/usr/local/include/eigen3
CERESLIBS = -L/usr/local/lib -lceres -lglog
CXXFLAGS = -Wall -ansi -pedantic -std=c++14 -O3 -funroll-loops -pipe $(CMPFITFLAGS) $(NLOPTFLAGS) $(CERESFLAGS)

TARGETS = least_squares

all: cmpfit $(TARGETS)

cmpfit:
	$(MAKE) -C cmpfit

clean:
	rm -f $(TARGETS) main.o

main.o : main.cpp *.hpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGETS) : main.o cmpfit/mpfit.o
	$(CXX) $(CXXFLAGS) -o $@ main.o $(CMPFITLIBS) $(NLOPTLIBS) $(CERESLIBS)
