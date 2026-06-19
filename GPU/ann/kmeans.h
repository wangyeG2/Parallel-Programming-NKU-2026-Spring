#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>

static float computeIPDistance(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return 1.0f - sum;
}

class KMeans {
public:
    size_t dim;
    int nlist;
    int max_iter;
    std::vector<float> centroids;

    KMeans(size_t dimension, int num_clusters, int iterations = 20) 
        : dim(dimension), nlist(num_clusters), max_iter(iterations) {
        centroids.resize(nlist * dim);
    }

    void fit(const float* data, size_t n) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist(0, n - 1);
        for (int i = 0; i < nlist; ++i) {
            size_t idx = dist(rng);
            memcpy(&centroids[i * dim], data + idx * dim, dim * sizeof(float));
        }

        std::vector<int> labels(n, -1);
        std::vector<float> new_centroids(nlist * dim, 0.0f);
        std::vector<int> cluster_sizes(nlist, 0);

        for (int iter = 0; iter < max_iter; ++iter) {
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 0; i < n; ++i) {
                float min_dist = std::numeric_limits<float>::max();
                int best_label = 0;
                for (int c = 0; c < nlist; ++c) {
                    float d = computeIPDistance(data + i * dim, &centroids[c * dim], dim);
                    if (d < min_dist) {
                        min_dist = d;
                        best_label = c;
                    }
                }
                labels[i] = best_label;
            }

            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);

            for (size_t i = 0; i < n; ++i) {
                int c = labels[i];
                cluster_sizes[c]++;
                for (size_t d = 0; d < dim; ++d) {
                    new_centroids[c * dim + d] += data[i * dim + d];
                }
            }

            for (int c = 0; c < nlist; ++c) {
                if (cluster_sizes[c] > 0) {
                    for (size_t d = 0; d < dim; ++d) {
                        new_centroids[c * dim + d] /= cluster_sizes[c];
                    }
                    float norm = 0.0f;
                    for (size_t d = 0; d < dim; ++d) {
                        norm += new_centroids[c * dim + d] * new_centroids[c * dim + d];
                    }
                    norm = std::sqrt(norm);
                    if (norm > 1e-6f) {
                        for (size_t d = 0; d < dim; ++d) {
                            new_centroids[c * dim + d] /= norm;
                        }
                    }
                    memcpy(&centroids[c * dim], &new_centroids[c * dim], dim * sizeof(float));
                }
            }
        }
    }
};

#endif // KMEANS_H
