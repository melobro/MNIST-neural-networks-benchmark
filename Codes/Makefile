# Define compiler
CXX=g++

# Define flags
CXXFLAGS= -fopenmp

# Target rules
all: train omp omp_simd omp_simd_bchmk test test_ll test_llb test_llb1

train: train.cpp
	$(CXX) train.cpp -o train

omp: omp.cpp
	$(CXX) $(CXXFLAGS) -o omp omp.cpp
	
omp_simd: omp_simd.cpp
	$(CXX) $(CXXFLAGS) -o omp_simd omp_simd.cpp

omp_simd_bchmk: omp_simd_bchmk.cpp
	$(CXX) $(CXXFLAGS) -o omp_simd_bchmk omp_simd_bchmk.cpp

test: test.cpp
	$(CXX) test.cpp -o test
	
test_ll: test_ll.cpp
	$(CXX) $(CXXFLAGS) -o test_ll test_ll.cpp
	
test_llb: test_llb.cpp
	$(CXX) $(CXXFLAGS) -o test_llb test_llb.cpp
	
test_llb1: test_ll.cpp
	$(CXX) $(CXXFLAGS) -o test_llb1 test_llb1.cpp
	

clean:
	rm -f train omp omp_simd omp_simd_bchmk test test_ll test_llb test_llb1
