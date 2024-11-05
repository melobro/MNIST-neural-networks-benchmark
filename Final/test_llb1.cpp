#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <chrono>
#include <omp.h> // Include OpenMP header

using namespace std;

const string testing_image_file = "mnist/t10k-images.idx3-ubyte";
const string testing_label_file = "mnist/t10k-labels.idx1-ubyte";
const string saved_weights = "model-neural-network.dat";
const string report_file = "testing-report_bchmk.dat";
const string benchmark_file = "benchmark_llb1.txt";

const int Training_samples = 10000;
const int width = 28;
const int height = 28;
const int input_neurons = width * height; 
const int hidden_neurons = 128; 
const int output_neurons = 10; 

ifstream image;
ifstream label;
ofstream report;
ofstream benchmark;

// Global variables for neural network layers and weights
double *weights_1[input_neurons + 1], *output_1;
double *weights_2[hidden_neurons + 1], *inputs_2, *outputs_2;
double *input_3, *output_3;
double expected[output_neurons + 1];
int d[width + 1][height + 1];

void init_array() {
    // Initialization of neural network arrays
    #pragma omp parallel for
    for (int i = 1; i <= input_neurons; ++i) {
        weights_1[i] = new double[hidden_neurons + 1];
    }
    
    output_1 = new double[input_neurons + 1];

    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
        weights_2[i] = new double[output_neurons + 1];
    }
    
    inputs_2 = new double[hidden_neurons + 1];
    outputs_2 = new double[hidden_neurons + 1];

    input_3 = new double[output_neurons + 1];
    output_3 = new double[output_neurons + 1];
}

void load_model(string file_name) {
    ifstream file(file_name.c_str(), ios::in);
    
    for (int i = 1; i <= input_neurons; ++i) { 
        for (int j = 1; j <= hidden_neurons; ++j) {
            file >> weights_1[i][j];
        }
    }
    
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            file >> weights_2[i][j];
        }
    }
    
    file.close();
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void perceptron() {
    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
        inputs_2[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 1; i <= output_neurons; ++i) {
        input_3[i] = 0.0;
    }

    #pragma omp parallel for
    for (int i = 1; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons; ++j) {
            inputs_2[j] += output_1[i] * weights_1[i][j];
        }
    }

    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
        outputs_2[i] = sigmoid(inputs_2[i]);
    }

    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            input_3[j] += outputs_2[i] * weights_2[i][j];
        }
    }

    #pragma omp parallel for
    for (int i = 1; i <= output_neurons; ++i) {
        output_3[i] = sigmoid(input_3[i]);
    }
}

double square_error() {
    double res = 0.0;
    #pragma omp parallel for reduction(+:res)
    for (int i = 1; i <= output_neurons; ++i) {
        res += (output_3[i] - expected[i]) * (output_3[i] - expected[i]);
    }
    res *= 0.5;
    return res;
}

int input() {
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
                d[i][j] = 0; 
            } else {
                d[i][j] = 1;
            }
        }
    }

    #pragma omp parallel for
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            output_1[pos] = d[i][j];
        }
    }

    label.read(&number, sizeof(char));
    for (int i = 1; i <= output_neurons; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    return (int)(number);
}

int main(int argc, char *argv[]) {

    image.open(testing_image_file.c_str(), ios::in | ios::binary);
    label.open(testing_label_file.c_str(), ios::in | ios::binary);
    report.open(report_file.c_str(), ios::out);
    benchmark.open(benchmark_file.c_str(), ios::out);

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    init_array();
    load_model(saved_weights);

    auto start_time_total = std::chrono::high_resolution_clock::now(); // Initialize the timer

    int nCorrect = 0;
    
    for (int sample = 1; sample <= Training_samples; ++sample) {
        int label = input();
        perceptron();
        
        int predict = 1;
        for (int i = 2; i <= output_neurons; ++i) {
            if (output_3[i] > output_3[predict]) {
                predict = i;
            }
        }
        --predict;

        double error = square_error();
        
        if (label == predict) {
            report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        } else {
            report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        }
    

        auto end_time_sample = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end_time_sample - start_time_total;
        benchmark << sample << " " << elapsed.count() << endl;
    }

    auto end_time_total = chrono::high_resolution_clock::now(); // End the timer
    chrono::duration<double> elapsed_total = end_time_total - start_time_total;

    image.close();
    label.close();
    report.close();
    benchmark.close();

    return 0;
}

