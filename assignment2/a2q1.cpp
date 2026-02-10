#include <iostream>
#include <vector>
#include <omp.h>
#include <cmath>
#include <iomanip>

using namespace std;

struct Particle { double x, y, z, fx, fy, fz; };

int main() {
    int N = 10000; 
    vector<Particle> p(N, {1.0, 2.0, 3.0, 0, 0, 0});
    double total_energy = 0.0;

    double serial_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double r2 = dx*dx + dy*dy + dz*dz + 1e-9; 

            if (r2 < 100.0) { 
                double r6 = r2 * r2 * r2;
                double force = 24.0 * (2.0 / (r6 * r6) - 1.0 / r6) / r2;
                total_energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6);
                p[i].fx += dx * force;
                p[j].fx -= dx * force;
                p[i].fy += dy * force;
                p[j].fy -= dy * force;
                p[i].fz += dz * force;
                p[j].fz -= dz * force;
            }
        }
    }
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    for (auto& particle : p) {
        particle.fx = particle.fy = particle.fz = 0.0;
    }
    total_energy = 0.0;
    double parallel_start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 16) reduction(+:total_energy)
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = p[i].x - p[j].x;
            double dy = p[i].y - p[j].y;
            double dz = p[i].z - p[j].z;
            double r2 = dx*dx + dy*dy + dz*dz + 1e-9; 

            if (r2 < 100.0) { 
                double r6 = r2 * r2 * r2;
                double force = 24.0 * (2.0 / (r6 * r6) - 1.0 / r6) / r2;
                total_energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6);

                #pragma omp atomic
                p[i].fx += dx * force;
                #pragma omp atomic
                p[j].fx -= dx * force;
                #pragma omp atomic
                p[i].fy += dy * force;
                #pragma omp atomic
                p[j].fy -= dy * force;
                #pragma omp atomic
                p[i].fz += dz * force;
                #pragma omp atomic
                p[j].fz -= dz * force;
            }
        }
    }
    double parallel_end = omp_get_wtime();
    double parallel_time = parallel_end - parallel_start;

    int num_threads = omp_get_max_threads();
    double speedup = serial_time / parallel_time;
    double efficiency = speedup / num_threads;
    cout << fixed << setprecision(4);
    cout << "MD Force Calculation Results:" << endl;
    cout << "Threads: " << num_threads << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    cout << "Efficiency: " << efficiency * 100 << "%" << endl;
    
    return 0;
}