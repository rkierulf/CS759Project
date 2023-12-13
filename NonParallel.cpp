#include <iostream>
#include <fstream>
#include <chrono>
#include <math.h>

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration;

double calc_y_prime(const double y, const double t)
{
    return -y * sin(t) - 0.1 * y;
}

double calc_y_exact(const double t)
{
    return exp(-0.1 * t + cos(t));
}

double euler_step(const double y, const double t, const double step_size)
{
    double y_prime = calc_y_prime(y, t);
    
    return y + y_prime * step_size;
}

void calc_solution(const double t_start, const double y_0, const double t_step, const unsigned int num_steps, double *t, double *y)
{
    t[0] = t_start;
    y[0] = y_0;

    for (unsigned int i = 1; i < num_steps; i++) {
        t[i] = t[i-1] + t_step;
        y[i] = euler_step(y[i-1], t[i-1], t_step);
    }
}

void write_output(const double *t, const double *y, const unsigned int num_steps, const double plot_resolution)
{
    ofstream fs;

    // Write t
    fs.open("data_t.txt");
    double t_current = 0.0;
    fs << t[0];
    for (unsigned int i = 1; i < num_steps; i++) {
        if (t[i] - t_current >= plot_resolution) {
            fs << "," << t[i];
            t_current = t[i];
        }
    }
    fs.close();

    // Write y
    fs.open("data_y.txt");
    t_current = 0.0;
    fs << y[0];
    for (unsigned int i = 1; i < num_steps; i++) {
        if (t[i] - t_current >= plot_resolution) {
            fs << "," << y[i];
            t_current = t[i];
        }   
    }
    fs.close();
}

int main(int argc, char* argv[])
{
    const double t_start = 0.0;                         // Time interval beginning
    const double t_end = 20.0;                          // Time interval end
    const double t_step = std::atof(argv[1]);           // Time step-size
    const double y_0 = exp(1.0);                        // Initial condition y(0)
    const double plot_resolution = 0.01;                // Resolution for plotting (only points this distance apart are included in output)
    high_resolution_clock::time_point start;            // Clock start
    high_resolution_clock::time_point end;              // Clock end
    duration<double, std::milli> duration_ms;           // Clock duration (ms)

    // Calculate number of time steps, initialize t and y
    const unsigned int num_steps = ceil((t_end - t_start) / t_step);
    double *t = new double[num_steps];
    double *y = new double[num_steps];

    // Integrate to end of time interval and record time taken
    start = high_resolution_clock::now();
    calc_solution(t_start, y_0, t_step, num_steps, t, y);
    end = high_resolution_clock::now();
    duration_ms = std::chrono::duration_cast<duration<double, std::milli>>(end - start); 

    // Write t and y to output files
    write_output(t, y, num_steps, plot_resolution);

    // Calculate error, print error and time taken
    double y_end_exact = calc_y_exact(t[num_steps-1]);
    printf("Error: %11.10lf\n", abs(y[num_steps-1] - y_end_exact));
    printf("Time (ms): %f\n", duration_ms.count());

    // Free t and y
    delete [] t;
    delete [] y;

    return 0;
}
