#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <random>

using namespace std;

string generate_dna_sequence(int length) {
    const char bases[] = {'A', 'T', 'C', 'G'};
    string seq;
    seq.reserve(length);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, 3);
    for (int i = 0; i < length; i++) {
        seq += bases[dis(gen)];
    }
    return seq;
}

int main() {
    int N = 5000, M = 5000;
    
    string seq1 = generate_dna_sequence(N);
    string seq2 = generate_dna_sequence(M); 
    vector<vector<int>> H(N + 1, vector<int>(M + 1, 0));

    int match = 2, mismatch = -1, gap = -1;

    double serial_start = omp_get_wtime();
    for (int k = 2; k <= (N + M); ++k) {
        int row_start = max(1, k - M);
        int row_end = min(k - 1, N);

        for (int i = row_start; i <= row_end; ++i) {
            int j = k - i;
            int score = (seq1[i-1] == seq2[j-1]) ? match : mismatch;
            H[i][j] = max({0, 
                           H[i-1][j-1] + score, 
                           H[i-1][j] + gap, 
                           H[i][j-1] + gap});
        }
    }
    double serial_end = omp_get_wtime();
    double serial_time = serial_end - serial_start;

    H.assign(N + 1, vector<int>(M + 1, 0));

    vector<string> schedules = {"static", "dynamic", "guided"};
    vector<double> times;
    
    for (const auto& sched : schedules) {
        H.assign(N + 1, vector<int>(M + 1, 0));
        
        double start = omp_get_wtime();
        for (int k = 2; k <= (N + M); ++k) {
            int row_start = max(1, k - M);
            int row_end = min(k - 1, N);

            if (sched == "static") {
                #pragma omp parallel for schedule(static)
                for (int i = row_start; i <= row_end; ++i) {
                    int j = k - i;
                    int score = (seq1[i-1] == seq2[j-1]) ? match : mismatch;
                    H[i][j] = max({0, 
                                   H[i-1][j-1] + score, 
                                   H[i-1][j] + gap, 
                                   H[i][j-1] + gap});
                }
            } else if (sched == "dynamic") {
                #pragma omp parallel for schedule(dynamic)
                for (int i = row_start; i <= row_end; ++i) {
                    int j = k - i;
                    int score = (seq1[i-1] == seq2[j-1]) ? match : mismatch;
                    H[i][j] = max({0, 
                                   H[i-1][j-1] + score, 
                                   H[i-1][j] + gap, 
                                   H[i][j-1] + gap});
                }
            } else if (sched == "guided") {
                #pragma omp parallel for schedule(guided)
                for (int i = row_start; i <= row_end; ++i) {
                    int j = k - i;
                    int score = (seq1[i-1] == seq2[j-1]) ? match : mismatch;
                    H[i][j] = max({0, 
                                   H[i-1][j-1] + score, 
                                   H[i-1][j] + gap, 
                                   H[i][j-1] + gap});
                }
            }
        }
        double end = omp_get_wtime();
        times.push_back(end - start);
    }
    
    double parallel_time = *min_element(times.begin(), times.end());

    int num_threads = omp_get_max_threads();
    double speedup = serial_time / parallel_time;
    double efficiency = speedup / num_threads;

    cout << fixed << setprecision(4);
    cout << "Sequence Alignment Results:" << endl;
    cout << "Threads: " << num_threads << endl;
    cout << "\nScheduling Strategy Performance:" << endl;
    for (size_t i = 0; i < schedules.size(); i++) {
        cout << "  " << schedules[i] << ": " << times[i] << "s" << endl;
    }
    cout << "\nBest Performance:" << endl;
    cout << "Speedup: " << speedup << "x" << endl;
    cout << "Efficiency: " << efficiency * 100 << "%" << endl;
    
    return 0;
}