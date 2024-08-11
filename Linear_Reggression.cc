//
// The purpose of this program is to implement a 
// simple linear regression model using the gradient
// descent algorithm.
//
// Author - Zain Khan

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// Function declarations
double hypothesis(double x, double m, double c);
double costFunction(const std::vector<double>& x, const std::vector<double>& y, double m, double c);
void gradientDescent(const std::vector<double>& x, const std::vector<double>& y, double& m, double& c, double learningRate, int iterations);
void writeDataToCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& predictions);

int main() {
    // Initialize data using push_back
    std::vector<double> x;
    x.push_back(1);
    x.push_back(2);
    x.push_back(3);
    x.push_back(4);
    x.push_back(5);

    std::vector<double> y;
    y.push_back(2);
    y.push_back(4);
    y.push_back(5);
    y.push_back(4);
    y.push_back(5);

    double m = 0;
    double c = 0;
    double learningRate = 0.01;
    int iterations = 1000;

    // Perform gradient descent
    gradientDescent(x, y, m, c, learningRate, iterations);

    // Display final values of m and c
    std::cout << "Final m: " << m << ", Final c: " << c << std::endl;

    // Display the final regression equation
    std::cout << "The final equation is: y = " << m << "x + " << c << std::endl;

    // Make predictions based on the final model
    std::vector<double> predictions;
    for (double val : x) {
        predictions.push_back(hypothesis(val, m, c));
    }

    // Output the predictions
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << "x = " << x[i] << ", Predicted y = " << predictions[i] << std::endl;
    }

    // Write the data and predictions to a CSV file for visualization
    writeDataToCSV(x, y, predictions);

    return 0;
}

// Function to calculate the hypothesis
double hypothesis(double x, double m, double c) {
    return m * x + c;
}

// Function to calculate the cost function (Mean Squared Error)
double costFunction(const std::vector<double>& x, const std::vector<double>& y, double m, double c) {
    double cost = 0;
    int n = x.size();
    for (int i = 0; i < n; ++i) {
        cost += pow(hypothesis(x[i], m, c) - y[i], 2);
    }
    return cost / (2 * n);
}

// Function to perform gradient descent
void gradientDescent(const std::vector<double>& x, const std::vector<double>& y, double& m, double& c, double learningRate, int iterations) {
    int n = x.size();
    for (int i = 0; i < iterations; ++i) {
        double sum_m = 0;
        double sum_c = 0;

        for (int j = 0; j < n; ++j) {
            double error = hypothesis(x[j], m, c) - y[j];
            sum_m += error * x[j];
            sum_c += error;
        }

        m -= learningRate * sum_m / n;
        c -= learningRate * sum_c / n;

        // Debugging print statements
        if (i % 100 == 0) {
            std::cout << "Iteration " << i << ": m = " << m << ", c = " << c 
                      << ", Cost = " << costFunction(x, y, m, c) << std::endl;
        }
    }
}

// Function to write data to a CSV file
void writeDataToCSV(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& predictions) {
    std::ofstream file("output.csv");
    file << "x,y,predictions\n";
    for (size_t i = 0; i < x.size(); ++i) {
        file << x[i] << "," << y[i] << "," << predictions[i] << "\n";
    }
    file.close();
}


