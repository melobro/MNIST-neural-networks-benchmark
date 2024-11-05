#!/bin/bash

# Get the start time
start_time=$(date +%s.%N)

# Compile and run the C++ code
./numbers

# Get the end time
end_time=$(date +%s.%N)

# Calculate the elapsed time
elapsed_time=$(echo "$end_time - $start_time" | bc)

# Save the elapsed time to a text file
echo "Elapsed time: $elapsed_time seconds" > elapsed_time.txt

echo "Elapsed time saved to elapsed_time.txt."

