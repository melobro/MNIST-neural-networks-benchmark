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
const string saved_weights = "model-neural-network-normal.dat";
const string report_file = "testing-report.dat";

const int Training_samples = 10000;
const int width = 28;
const int height = 28;
const int input_neurons = width * height; 
const int hidden_neurons = 128; 
const int output_neurons = 10; 

// Input layer - Hidden layer
double *weights_1[input_neurons + 1], *output_1;

// Hidden layer - Output layer
double *weights_2[hidden_neurons + 1], *inputs_2, *outputs_2;

// Output layer
double *input_3, *output_3;
double expected[output_neurons + 1];

int d[width + 1][height + 1];

ifstream image;
ifstream label;
ofstream report;

void init_array() {
    // Input layer - Hidden layer
    #pragma omp parallel for
    for (int i = 1; i <= input_neurons; ++i) {
        weights_1[i] = new double[hidden_neurons + 1];
    }
    
    output_1 = new double[input_neurons + 1];

    // Hidden layer - Output layer
    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
        weights_2[i] = new double[output_neurons + 1];
    }
    
    inputs_2 = new double[hidden_neurons + 1];
    outputs_2 = new double[hidden_neurons + 1];

    // Output layer
    input_3 = new double[output_neurons + 1];
    output_3 = new double[output_neurons + 1];
}

void load_model(string file_name) {
    ifstream file(file_name.c_str(), ios::in);
    
    // Input layer - Hidden layer
//    #pragma omp parallel for
    for (int i = 1; i <= input_neurons; ++i) { 
        for (int j = 1; j <= hidden_neurons; ++j) {
            file >> weights_1[i][j];
        }
    }
    
    // Hidden layer - Output layer
//    #pragma omp parallel for
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
    #pragma omp simd
        for (int j = 1; j <= hidden_neurons; ++j) {
            inputs_2[j] += output_1[i] * weights_1[i][j];
        }
    }

    #pragma omp simd
    for (int i = 1; i <= hidden_neurons; ++i) {
        outputs_2[i] = sigmoid(inputs_2[i]);
    }

    #pragma omp parallel for
    for (int i = 1; i <= hidden_neurons; ++i) {
    #pragma omp simd
        for (int j = 1; j <= output_neurons; ++j) {
            input_3[j] += outputs_2[i] * weights_2[i][j];
        }
    }

    #pragma omp simd
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
    // Reading image
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
    #pragma omp simd
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            output_1[pos] = d[i][j];
        }
    }

    // Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= output_neurons; ++i) {
        expected[i] = 0.0;
    }
    expected[number + 1] = 1.0;

    return (int)(number);
}

int main(int argc, char *argv[]) {
    report.open(report_file.c_str(), ios::out);
    image.open(testing_image_file.c_str(), ios::in | ios::binary);
    label.open(testing_label_file.c_str(), ios::in | ios::binary);

    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    // NN Initialization
    init_array();
    load_model(saved_weights);

    auto start_time_total = std::chrono::high_resolution_clock::now(); // Initialize the timer

    int nCorrect = 0;
    
    for (int sample = 1; sample <= Training_samples; ++sample) {
        cout << "Sample " << sample << endl;

        int label = input();

        perceptron();
        
        // Prediction
        int predict = 1;
        for (int i = 2; i <= output_neurons; ++i) {
            if (output_3[i] > output_3[predict]) {
                predict = i;
            }
        }
        --predict;

        // Writing down
        double error = square_error();
        printf("Error: %0.6lf\n", error);
        
        if (label == predict) {
            ++nCorrect;
            cout << "Classification: YES. Label = " << label << ". Predict = " << predict << endl << endl;
            report << "Sample " << sample << ": YES. Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        } else {
            cout << "Classification: NO.  Label = " << label << ". Predict = " << predict << endl << endl;
            report << "Sample " << sample << ": NO.  Label = " << label << ". Predict = " << predict << ". Error = " << error << endl;
        }
    }
    
    auto end_time_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time_total - start_time_total;
    cout << "Time taken to process all the samples: " << elapsed.count() << " seconds." << endl;

    // Summary
    double accuracy = (double)(nCorrect) / Training_samples * 100.0;
    cout << "Number of correct samples: " << nCorrect << " / " << Training_samples << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << nCorrect << " / " << Training_samples << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();

    return 0;
 }
