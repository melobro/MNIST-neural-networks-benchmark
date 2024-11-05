# Define compiler
CXX=g++

# Define flags
CXXFLAGS= -fopenmp

# Define shell scripts
SHELL_SCRIPT1=bench_test.sh
SHELL_SCRIPT2=bench_train.sh

# Target rules for C++ compilation
all: omp_simd omp_simd_bchmk test_ll test_llb1 check_script

omp_simd: omp_simd.cpp
	$(CXX) $(CXXFLAGS) -o omp_simd omp_simd.cpp

omp_simd_bchmk: omp_simd_bchmk.cpp
	$(CXX) $(CXXFLAGS) -o omp_simd_bchmk omp_simd_bchmk.cpp

test_ll: test_ll.cpp
	$(CXX) $(CXXFLAGS) -o test_ll test_ll.cpp

test_llb1: test_llb1.cpp
	$(CXX) $(CXXFLAGS) -o test_llb1 test_llb1.cpp

# Target rules for shell script checking
check_script:
	@echo "\nChecking shell scripts for syntax errors..."
	@bash -n $(SHELL_SCRIPT1) && echo "No syntax errors in $(SHELL_SCRIPT1)." || echo "Syntax errors found in $(SHELL_SCRIPT1)."
	@bash -n $(SHELL_SCRIPT2) && echo "No syntax errors in $(SHELL_SCRIPT2)." || echo "Syntax errors found in $(SHELL_SCRIPT2)."

clean:
	rm -f omp_simd omp_simd_bchmk test_ll test_llb1

.PHONY: all check_script clean

