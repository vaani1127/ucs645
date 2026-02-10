#include <iostream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <algorithm>

using namespace std;

void solve_heat() {
    int N = 2048; 
    int steps = 100;
    int TILE_SIZE = 32; 
    vector<double> curr(N*N, 20.0), next(N*N, 20.0);

    for(int i=0; i<N; i++) curr[i] = 100.0; 

    double serial_start = omp_get_wtime();
    double total_heat_serial = 0.0;
    for (int t = 0; t < steps; t++) {
        for (int i = 1; i < N-1; i++) {
            for (int j = 1; j < N-1; j++) {
                next[i*N+j] = 0.25 * (curr[(i-1)*N+j] + curr[(i+1)*N+j] + 
                                      curr[i*N+(j-1)] + curr[i*N+(j+1)]);
            }
        }
        for(int i=0; i<N*N; i++) curr[i] = next[i]; 
    }
    for(int i=0; i<N*N; i++) total_heat_serial += curr[i];
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    vector<string> schedules = {"static", "dynamic-16", "guided"};
    vector<double> times;
    double total_heat_parallel = 0.0;

    for (const auto& sched : schedules) {
        curr.assign(N*N, 20.0);
        next.assign(N*N, 20.0);
        for(int i=0; i<N; i++) curr[i] = 100.0; 

        double start = omp_get_wtime();
        for (int t = 0; t < steps; t++) {
            if (sched == "static") {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int ii = 1; ii < N-1; ii += TILE_SIZE) {
                    for (int jj = 1; jj < N-1; jj += TILE_SIZE) {
                        for (int i = ii; i < min(ii + TILE_SIZE, N-1); i++) {
                            for (int j = jj; j < min(jj + TILE_SIZE, N-1); j++) {
                                next[i*N+j] = 0.25 * (curr[(i-1)*N+j] + curr[(i+1)*N+j] + 
                                                      curr[i*N+(j-1)] + curr[i*N+(j+1)]);
                            }
                        }
                    }
                }
            } else if (sched == "dynamic-16") {
                #pragma omp parallel for collapse(2) schedule(dynamic, 16)
                for (int ii = 1; ii < N-1; ii += TILE_SIZE) {
                    for (int jj = 1; jj < N-1; jj += TILE_SIZE) {
                        for (int i = ii; i < min(ii + TILE_SIZE, N-1); i++) {
                            for (int j = jj; j < min(jj + TILE_SIZE, N-1); j++) {
                                next[i*N+j] = 0.25 * (curr[(i-1)*N+j] + curr[(i+1)*N+j] + 
                                                      curr[i*N+(j-1)] + curr[i*N+(j+1)]);
                            }
                        }
                    }
                }
            } else if (sched == "guided") {
                #pragma omp parallel for collapse(2) schedule(guided)
                for (int ii = 1; ii < N-1; ii += TILE_SIZE) {
                    for (int jj = 1; jj < N-1; jj += TILE_SIZE) {
                        for (int i = ii; i < min(ii + TILE_SIZE, N-1); i++) {
                            for (int j = jj; j < min(jj + TILE_SIZE, N-1); j++) {
                                next[i*N+j] = 0.25 * (curr[(i-1)*N+j] + curr[(i+1)*N+j] + 
                                                      curr[i*N+(j-1)] + curr[i*N+(j+1)]);
                            }
                        }
                    }
                }
            }
            
            #pragma omp parallel for
            for(int i=0; i<N*N; i++) curr[i] = next[i];
        }
        
        total_heat_parallel = 0.0;
        #pragma omp parallel for reduction(+:total_heat_parallel)
        for(int i=0; i<N*N; i++) {
            total_heat_parallel += curr[i];
        }
        
        double end = omp_get_wtime();
        times.push_back(end - start);
    }

    double parallel_time = *min_element(times.begin(), times.end());

    int num_threads = omp_get_max_threads();
    double speedup = serial_time / parallel_time;
    double efficiency = speedup / num_threads;
    double data_moved = (double)steps * 4.0 * N * N * sizeof(double) / 1e9; // GB

    cout << fixed << setprecision(4);
    cout << "Heat Equation Solver Results:" << endl;
    cout << "Threads: " << num_threads << endl;
    cout << "\nScheduling Strategy Performance:" << endl;
    for (size_t i = 0; i < schedules.size(); i++) {
        cout << "  " << schedules[i] << ": " << times[i] << "s" << endl;
    }
    cout << "\nBest Performance:" << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    cout << "Efficiency: " << efficiency * 100 << "%" << endl;
    cout << "Bandwidth: " << data_moved/parallel_time << " GB/s" << endl;
    cout << "Total Heat (final): " << total_heat_parallel << endl;
}

int main() { solve_heat(); return 0; }
