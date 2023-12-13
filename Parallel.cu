#include <iostream>
#include <fstream>
#include <math.h>
#include <cuda.h>

using namespace std;

double calc_y_exact(const double t)
{
    return exp(-0.1 * t + cos(t));
}

double y_initial_guess(const double t, const double y_0)
{
    return y_0;
}

void calc_interval_sizes(const double t_start, const double t_end, const double t_step, const unsigned int num_intervals, double *t_interval_starts, int *interval_steps)
{
    unsigned int num_interval_steps = floor((t_end - t_start) / (num_intervals * t_step));

    for (unsigned int i = 0; i < num_intervals - 1; i++) {
        t_interval_starts[i] = i * num_interval_steps * t_step;
        interval_steps[i] = num_interval_steps;
    }

    // Last interval may be slightly longer
    t_interval_starts[num_intervals-1] = (num_intervals - 1) * num_interval_steps * t_step;
    interval_steps[num_intervals-1] = ceil((t_end - t_interval_starts[num_intervals-1]) / t_step);
}

void calc_y_interval_starts(const double *t_interval_starts, const unsigned int num_intervals, const double y_0, double *y_interval_starts)
{
    for (unsigned int i = 0; i < num_intervals; i++) {
        y_interval_starts[i] = y_initial_guess(t_interval_starts[i], y_0);
    }
}

void write_output(double ***t_ref, double ***y_ref, int *interval_steps, const unsigned int num_intervals, const double plot_resolution)
{
    ofstream fs;
    double **t = *t_ref;
    double **y = *y_ref;

    // Write t
    fs.open("data_t.txt");
    double t_current = 0.0;
    fs << t[0][0];
    for (unsigned int i = 0; i < num_intervals; i++) {
        for (int j = 0; j < interval_steps[i]; j++) {
            if (t[i][j] - t_current >= plot_resolution) {
                fs << "," << t[i][j];
                t_current = t[i][j];
            }
        }
    }
    fs.close();

    // Write y
    fs.open("data_y.txt");
    t_current = 0.0;
    fs << y[0][0];
    for (unsigned int i = 0; i < num_intervals; i++) {
        for (int j = 0; j < interval_steps[i]; j++) {
            if (t[i][j] - t_current >= plot_resolution) {
                fs << "," << y[i][j];
                t_current = t[i][j];
            }
        }
    }
    fs.close();
}

__global__ void fill_t(double *t_interval_starts, double **t, int *interval_steps, unsigned int num_intervals, double t_step)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_intervals) {
        return;
    }

    int num_steps = interval_steps[tid];
    double t_start = t_interval_starts[tid];
    t[tid][0] = t_start;

    for (int i = 1; i < num_steps+1; i++) {
        t[tid][i] = t_start + i * t_step;
    }
}

__global__ void integrate_kernel(double *t_interval_starts, double *y_interval_starts, int *interval_steps, double **t, double **y, double **dy, unsigned int num_intervals, double t_step, double eps)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_intervals * 2) {
        return;
    }

    int num_steps = interval_steps[tid / 2];
    double y_start = y_interval_starts[tid / 2];
    double *t_sol = t[tid / 2];
    double *y_sol;
    
    if (tid % 2 == 0) {
        y_sol = y[tid / 2];
        y_sol[0] = y_start;
    }
    else {
        y_sol = dy[tid / 2];
        y_sol[0] = y_start + eps;
    }

    for (int i = 1; i < num_steps+1; i++) {
        y_sol[i] = y_sol[i-1] + t_step * (-y_sol[i-1] * sin(t_sol[i-1]) - 0.1 * y_sol[i-1]);
    }
}

__host__ void prepare_update_calculation(double *dl, double *F, double*** y_ref, double ***dy_ref, int num_intervals, int *interval_steps, double eps)
{
    double **y = *y_ref;
    double **dy = *dy_ref;

    // Calculate F
    for (int i = 0; i < num_intervals - 1; i++) {
        F[i] = y[i+1][0] - y[i][interval_steps[i]]; // End of previous interval minus beginning of next interval,
    }

    // Calculate dl
    for (int i = 0; i < num_intervals - 2; i++) {
        dl[i] = (dy[i+1][interval_steps[i+1]] - y[i+1][interval_steps[i+1]]) / eps; // Finite difference calculation
    }
}

__host__ void calc_update(double *dl, double *F, double *dx, unsigned int num_intervals, double *update_size)
{
    *update_size = 0.0;

    // Specialized bidiagonal algorithm: main diagonal is -1s, upper diagonal is 0s
    dx[0] = -F[0];
    for (unsigned int i = 1; i < num_intervals-1; i++) {
        dx[i] = dl[i-1] * dx[i-1] - F[i];
        *update_size += abs(dx[i]);
    }

    *update_size = (*update_size) / (num_intervals - 1);
}

__host__ void calc_solution(double *t_interval_starts, double *y_interval_starts, int *interval_steps, double ***t_ref, double ***y_ref, double ***dy_ref, unsigned int num_intervals, unsigned int block_size, double t_step, double eps, double tol)
{
    double **t = *t_ref;
    double **y = *y_ref;
    double **dy = *dy_ref;

    unsigned int num_blocks = (num_intervals + block_size - 1) / block_size;
    fill_t<<<num_blocks, block_size>>>(t_interval_starts, t, interval_steps, num_intervals, t_step);
    cudaDeviceSynchronize();

    bool convergence_reached = false;
    num_blocks = ((num_intervals + block_size - 1) / block_size) * 2;
    unsigned int num_iterations = 0;

    double *dl = new double[num_intervals-2];    // Jacobian lower diagonal
    double *F = new double[num_intervals-1];     // Function values
    double *dx = new double[num_intervals-1];    // Interval start update, solved for in Jacobian update iteration
    double update_size = 0.0;

    for (unsigned int i = 0; i < num_intervals; i++) {
        y[i][0] = y_interval_starts[i];
        dy[i][0] = y_interval_starts[i];
    }

    while (convergence_reached == false) {
        integrate_kernel<<<num_blocks, block_size>>>(t_interval_starts, y_interval_starts, interval_steps, t, y, dy, num_intervals, t_step, eps);
        cudaDeviceSynchronize();
        prepare_update_calculation(dl, F, y_ref, dy_ref, num_intervals, interval_steps, eps);
        calc_update(dl, F, dx, num_intervals, &update_size);
        
        // Make update
        for (unsigned int i = 1; i < num_intervals; i++) {
            y_interval_starts[i] += dx[i-1];
        }
        num_iterations += 1;

        if (update_size <= tol) {
            convergence_reached = true;
            printf("Converged in %d iterations \n", num_iterations);
        }

        // Algorithm should converge in a few iterations
        if (num_iterations == 100 && convergence_reached == false) {
            convergence_reached = true;
            printf("Failed to converge after 100 iterations \n");
        }
    }

    delete [] dl;
    delete [] F;
    delete [] dx;
}

int main(int argc, char* argv[])
{
    const double t_start = 0.0;                                   // Time interval beginning
    const double t_end = 20.0;                                    // Time interval end
    const double t_step = std::atof(argv[1]);                     // Time step-size
    const unsigned int num_intervals = std::atoi(argv[2]);        // Number of time subintervals
    const unsigned int block_size = std::atoi(argv[3]);           // Threads per block (for integrate_kernel)
    const double y_0 = exp(1.0);                                  // Initial condition y(0)
    const double plot_resolution = 0.01;                          // Resolution for plotting (only points this distance apart are included in output)
    const double eps = t_step / 100.0;                            // Small number, used for finite difference calculation of Jacobian terms
    const double tol = 0.00000000001;                             // Tolerence to determine when convergence reached
    cudaEvent_t start;                                            // Events for timing            
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float duration;

    // Calculate interval start values and allocate memory
    double *t_interval_starts;
    double *y_interval_starts;
    int *interval_steps;
    cudaMallocManaged(&t_interval_starts, num_intervals * sizeof(double));
    cudaMallocManaged(&y_interval_starts, num_intervals * sizeof(double));
    cudaMallocManaged(&interval_steps, num_intervals * sizeof(int));
    calc_interval_sizes(t_start, t_end, t_step, num_intervals, t_interval_starts, interval_steps);
    calc_y_interval_starts(t_interval_starts, num_intervals, y_0, y_interval_starts);
    double **t;
    double **y;
    double **dy;
    cudaMallocManaged(&t, num_intervals * sizeof(double *));
    cudaMallocManaged(&y, num_intervals * sizeof(double *));
    cudaMallocManaged(&dy, num_intervals * sizeof(double *));
    for (unsigned int i = 0; i < num_intervals; i++) {
        cudaMallocManaged(&t[i], (interval_steps[i] + 1) * sizeof(double));
        cudaMallocManaged(&y[i], (interval_steps[i] + 1) * sizeof(double));
        cudaMallocManaged(&dy[i], (interval_steps[i] + 1) * sizeof(double));
    }
    
    // Main computation
    cudaEventRecord(start);
    calc_solution(t_interval_starts, y_interval_starts, interval_steps, &t, &y, &dy, num_intervals, block_size, t_step, eps, tol);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&duration, start, end);

    // Print accuracy and time taken
    printf("Error: %18.15f \n", abs(calc_y_exact(t_end) - y[num_intervals-1][interval_steps[num_intervals-1]]));
    printf("Time(ms): %f \n", duration);

    // Write t and y to output files
    write_output(&t, &y, interval_steps, num_intervals, plot_resolution); 

    // Free variables
    for (unsigned int i = 0; i < num_intervals; i++) {
        cudaFree(t[i]);
        cudaFree(y[i]);
        cudaFree(dy[i]);
    }
    cudaFree(t);
    cudaFree(y);
    cudaFree(dy);
    cudaFree(t_interval_starts);
    cudaFree(y_interval_starts);
    cudaFree(interval_steps);

    return 0;
}
