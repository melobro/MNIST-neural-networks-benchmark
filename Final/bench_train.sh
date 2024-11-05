#!/bin/bash

benchmark_folder="benchmark_results_train"
mkdir -p $benchmark_folder

g++ -o omp_simd_bchmk -fopenmp omp_simd_bchmk.cpp

# Loop over different number of threads
for threads in 1 2 4 8
do
    export OMP_NUM_THREADS=$threads
    echo "Running with Threads=$threads"
    
    ./omp_simd_bchmk
  
    cp benchmark_train.txt $benchmark_folder/bchmk_train_t.no.$threads.txt
done

echo "Benchmarking completed. Results saved in corresponding files"

