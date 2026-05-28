#ifndef IVF_HNSW_H
#define IVF_HNSW_H

#include "hnswlib/hnswlib/hnswlib.h"
#include "kmeans.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <sys/stat.h>

class IVF_HNSW_Index {
public:
    size_t dim;
    int nlist;
    int M;
    int ef_construction;
    
    KMeans quantizer;
    hnswlib::InnerProductSpace* ipspace; // 修复：提升为类成员，在堆上分配
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_indices;

    IVF_HNSW_Index(size_t dimension, int num_clusters, int m, int ef_constr) 
        : dim(dimension), nlist(num_clusters), M(m), ef_construction(ef_constr), quantizer(dimension, num_clusters) {
        ipspace = new hnswlib::InnerProductSpace(dim); // 修复：确保生命周期与索引一致
    }

    ~IVF_HNSW_Index() {
        for (auto* idx : hnsw_indices) {
            if(idx) delete idx;
        }
        delete ipspace; // 回收内存
    }

    void create_directory(const std::string& path) {
        mkdir(path.c_str(), 0777);
    }

    void build_and_save(float* base, size_t n, const std::string& index_path) {
        create_directory(index_path);
        
        std::cout << "Training KMeans..." << std::endl;
        quantizer.fit(base, n);

        std::ofstream out(index_path + "/centroids.bin", std::ios::binary);
        out.write((char*)quantizer.centroids.data(), nlist * dim * sizeof(float));
        out.close();

        std::vector<std::vector<float>> clusters_data(nlist);
        std::vector<std::vector<uint32_t>> clusters_ids(nlist);

        for (size_t i = 0; i < n; ++i) {
            float min_dist = std::numeric_limits<float>::max();
            int best_cluster = 0;
            for (int c = 0; c < nlist; ++c) {
                float d = computeIPDistance(base + i * dim, &quantizer.centroids[c * dim], dim);
                if (d < min_dist) {
                    min_dist = d;
                    best_cluster = c;
                }
            }
            clusters_data[best_cluster].insert(
                clusters_data[best_cluster].end(), 
                base + i * dim, 
                base + (i + 1) * dim
            );
            clusters_ids[best_cluster].push_back(i);
        }

        std::cout << "Building HNSW for each cluster..." << std::endl;
        hnsw_indices.resize(nlist, nullptr);

        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < nlist; ++c) {
            size_t cluster_size = clusters_ids[c].size();
            if (cluster_size == 0) continue;
            
            // 修复：使用类成员 ipspace
            hnswlib::HierarchicalNSW<float>* appr_alg = new hnswlib::HierarchicalNSW<float>(
                ipspace, cluster_size, M, ef_construction
            );

            appr_alg->addPoint(clusters_data[c].data(), clusters_ids[c][0]);

            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 1; i < cluster_size; ++i) {
                appr_alg->addPoint(&clusters_data[c][i * dim], clusters_ids[c][i]);
            }
            
            appr_alg->saveIndex(index_path + "/cluster_" + std::to_string(c) + ".bin");
            hnsw_indices[c] = appr_alg;
        }
        std::cout << "Index build and save complete." << std::endl;
    }

    void load(const std::string& index_path) {
        quantizer.centroids.resize(nlist * dim);
        std::ifstream in(index_path + "/centroids.bin", std::ios::binary);
        in.read((char*)quantizer.centroids.data(), nlist * dim * sizeof(float));
        in.close();

        hnsw_indices.resize(nlist, nullptr);

        for (int c = 0; c < nlist; ++c) {
            std::string filename = index_path + "/cluster_" + std::to_string(c) + ".bin";
            std::ifstream file_check(filename);
            if (file_check.good()) {
                // 修复：使用类成员 ipspace，避免悬空指针
                hnsw_indices[c] = new hnswlib::HierarchicalNSW<float>(ipspace, filename);
            }
        }
        std::cout << "Index loaded from disk." << std::endl;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, int nprobe, int ef_search) 
    {
        std::vector<std::pair<float, int>> centroid_dists(nlist);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < nlist; ++c) {
            float dist = computeIPDistance(query, &quantizer.centroids[c * dim], dim);
            centroid_dists[c] = {dist, c};
        }

        std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end());

        std::vector<std::vector<std::pair<float, uint32_t>>> local_results(nprobe);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < nprobe; ++i) {
            int cluster_id = centroid_dists[i].second;
            if (hnsw_indices[cluster_id] == nullptr) continue;

            hnsw_indices[cluster_id]->ef_ = ef_search;
            auto res = hnsw_indices[cluster_id]->searchKnn(query, k);
            
            local_results[i].resize(res.size());
            int sz = res.size();
            while (!res.empty()) {
                local_results[i][--sz] = res.top();
                res.pop();
            }
        }

        std::vector<std::pair<float, uint32_t>> all_results;
        for (int i = 0; i < nprobe; ++i) {
            all_results.insert(all_results.end(), local_results[i].begin(), local_results[i].end());
        }

        size_t topk = std::min(k, all_results.size());
        std::partial_sort(all_results.begin(), all_results.begin() + topk, all_results.end());

        std::priority_queue<std::pair<float, uint32_t>> final_res;
        for (size_t i = 0; i < topk; ++i) {
            final_res.push(all_results[i]);
        }
        return final_res;
    }
};

#endif // IVF_HNSW_H
