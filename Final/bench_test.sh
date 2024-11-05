#!/bin/bash

benchmark_folder="benchmark_results_llb1"
mkdir -p $benchmark_folder

g++ -o test_llb1 -fopenmp test_llb1.cpp

# Loop over different number of threads
for threads in 1 2 4 8
do
    export OMP_NUM_THREADS=$threads
    echo "Running with Threads=$threads"

    ./test_llb1
  
    cp benchmark_llb1.txt $benchmark_folder/bchmk_llb1_t.no.$threads.txt
done

echo "Benchmarking completed. Results saved in $benchmark_folder"

