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
#include <omp.h> // Include OpenMP header
#include <chrono>

using namespace std;

const string training_image_file = "mnist/train-images.idx3-ubyte";
const string training_label_file = "mnist/train-labels.idx1-ubyte";
const string saved_weights = "model-neural-network.dat";
const string report_file = "training-report.dat";

const int Training_samples = 60000;
const int width = 28;
const int height = 28;

const int input_neurons = width * height; 
const int hidden_neurons = 128; 
const int output_neurons = 10; 
const int epochs = 550;
const double learning_rate = 1e-3;
const double momentum = 0.9;
const double error_threshold = 1e-3;

double *weights_1[input_neurons + 1], *corrections_1[input_neurons + 1], *output_1;
double *weights_2[hidden_neurons + 1], *corrections_2[hidden_neurons + 1], *inputs_2, *outputs_2, *errors_2;
double *input_3, *output_3, *error_3;
double expected[output_neurons + 1];

int d[width + 1][height + 1];

ifstream image;
ifstream label;
ofstream report;

void init_array() {
    // Input layer - Hidden layer
    for (int i = 1; i <= input_neurons; ++i) {
        weights_1[i] = new double[hidden_neurons + 1];
        corrections_1[i] = new double[hidden_neurons + 1];
                 }

    output_1 = new double[input_neurons + 1];

    // Hidden layer - Output layer
    for (int i = 1; i <= hidden_neurons; ++i) {
        weights_2[i] = new double[output_neurons + 1];
        corrections_2[i] = new double[output_neurons + 1];
          }

    inputs_2 = new double[hidden_neurons + 1];
    outputs_2 = new double[hidden_neurons + 1];
    errors_2 = new double[hidden_neurons + 1];

    // Output layer
    input_3 = new double[output_neurons + 1];
     output_3 = new double[output_neurons + 1];
     error_3 = new double[output_neurons + 1];

    // Initialization 1
    for (int i = 1; i <= input_neurons; ++i) {
         for (int j = 1; j <= hidden_neurons; ++j) {
            int sign = rand() % 2;

            weights_1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
                weights_1[i][j] = -weights_1[i][j];
                 }
                   }
            }

    // Initialization 2
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            int sign = rand() % 2;

            weights_2[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
                weights_2[i][j] = -weights_2[i][j];
            }
            }
             }
     }

// Our activation function
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

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons; ++j) {
            inputs_2[j] += output_1[i] * weights_1[i][j];
        }
    }

    for (int i = 1; i <= hidden_neurons; ++i) {
        outputs_2[i] = sigmoid(inputs_2[i]);
       }

    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            input_3[j] += outputs_2[i] * weights_2[i][j];
        }
          }

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

void back_propagation() {
    #pragma omp parallel for
    for (int i = 1; i <= output_neurons; ++i) {
        error_3[i] = output_3[i] * (1 - output_3[i]) * (expected[i] - output_3[i]);
    }

    double sum;
    #pragma omp parallel for private(sum)
    for (int i = 1; i <= hidden_neurons; ++i) {
        sum = 0.0;
        for (int j = 1; j <= output_neurons; ++j) {
            sum += weights_2[i][j] * error_3[j];
        }
        errors_2[i] = outputs_2[i] * (1 - outputs_2[i]) * sum;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            corrections_2[i][j] = (learning_rate * error_3[j] * outputs_2[i]) + (momentum * corrections_2[i][j]);
            weights_2[i][j] += corrections_2[i][j];
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons; ++j) {
            corrections_1[i][j] = (learning_rate * errors_2[j] * output_1[i]) + (momentum * corrections_1[i][j]);
            weights_1[i][j] += corrections_1[i][j];
        }
    }
}

int learning_process() {
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons; ++j) {
            corrections_1[i][j] = 0.0;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            corrections_2[i][j] = 0.0;
        }
    }

    for (int i = 1; i <= epochs; ++i) {
        perceptron();
        back_propagation();
        if (square_error() < error_threshold) {
            return i;
        }
    }
    return epochs;
}


void input() {
    char number;

    //#pragma omp parallel for collapse(2)
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

    #pragma omp parallel for collapse(2)
    for (int j = 1; j <= height; ++j) {
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

    cout << "Label: " << (int)(number) << endl;
}

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);

    if (!file.is_open()) {
        cerr << "Error opening file: " << file_name << endl;
        return;
       }

    // Input layer - Hidden layer
    for (int i = 1; i <= input_neurons; ++i) {
        for (int j = 1; j <= hidden_neurons; ++j) {
            file << weights_1[i][j] << " ";
        }
        file << endl;
       }

    // Hidden layer - Output layer
    for (int i = 1; i <= hidden_neurons; ++i) {
        for (int j = 1; j <= output_neurons; ++j) {
            file << weights_2[i][j] << " ";
        }
        file << endl;
    }

    file.close();
   }


int main(int argc, char *argv[]) {
    report.open(report_file.c_str(), ios::out);
    image.open(training_image_file.c_str(), ios::in | ios::binary);
    label.open(training_label_file.c_str(), ios::in | ios::binary);
    
    init_array();
        
    // Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
    }
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
    }

    auto start_time_total = std::chrono::high_resolution_clock::now(); // Initialize the timers
    auto start_time = std::chrono::high_resolution_clock::now(); 

    for (int sample = 1; sample <= Training_samples; ++sample) {
        if (sample % 100 == 1) { // Reset the timer for each new batch of 100 samples
            start_time = std::chrono::high_resolution_clock::now();
        }

        cout << "Sample " << sample << endl;

        input();

        int nIterations = learning_process();

        cout << "No. iterations: " << nIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ": No. iterations = " << nIterations << ", Error = " << square_error() << endl;

        if (sample % 100 == 0) {
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            cout << "Time taken for last 100 samples: " << elapsed.count() << " seconds." << endl;

            cout << "Saving the network to " << saved_weights << " file" << endl << endl;
            write_matrix(saved_weights);
        }
    }
    
    write_matrix(saved_weights);

            auto end_time_total = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time_total - start_time_total;
            cout << "Time taken to process all the samples: " << elapsed.count() << " seconds." << endl;

    report.close();
    image.close();
    label.close();

    return 0;
}

