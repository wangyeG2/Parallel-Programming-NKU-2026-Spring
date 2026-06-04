#ifndef IVF_HNSW_MPI_H
#define IVF_HNSW_MPI_H

#include "hnswlib/hnswlib/hnswlib.h"
#include "kmeans.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <sys/stat.h>
#include <mpi.h>

class IVF_HNSW_MPI_Index {
public:
    size_t dim;
    int nlist;
    int M;
    int ef_construction;
    KMeans quantizer;
    hnswlib::InnerProductSpace* ipspace;
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_indices;
    int rank;
    int size;
    std::vector<int> my_clusters;
    int start_cluster; // 记录负责的起始簇
    int end_cluster;   // 记录负责的结束簇

    IVF_HNSW_MPI_Index(size_t dimension, int num_clusters, int m, int ef_constr, 
                       int mpi_rank, int mpi_size) 
        : dim(dimension), nlist(num_clusters), M(m), ef_construction(ef_constr),
          quantizer(dimension, num_clusters), rank(mpi_rank), size(mpi_size) {
        ipspace = new hnswlib::InnerProductSpace(dim);
        
        int clusters_per_rank = nlist / size;
        int remainder = nlist % size;
        start_cluster = rank * clusters_per_rank + std::min(rank, remainder);
        end_cluster = start_cluster + clusters_per_rank + (rank < remainder ? 1 : 0);
        
        for (int c = start_cluster; c < end_cluster; ++c) {
            my_clusters.push_back(c);
        }
    }

    ~IVF_HNSW_MPI_Index() {
        for (auto* idx : hnsw_indices) {
            if(idx) delete idx;
        }
        delete ipspace;
    }

    void create_directory(const std::string& path) {
        if (rank == 0) {
            mkdir(path.c_str(), 0777);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void build_and_save(float* base, size_t n, const std::string& index_path) {
        create_directory(index_path);

        if (rank == 0) {
            std::cout << "Training KMeans..." << std::endl;
            quantizer.fit(base, n);
            
            std::ofstream out(index_path + "/centroids.bin", std::ios::binary);
            out.write((char*)quantizer.centroids.data(), nlist * dim * sizeof(float));
            out.close();
        }

        MPI_Bcast(quantizer.centroids.data(), nlist * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

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
                base + i * dim, base + (i + 1) * dim
            );
            clusters_ids[best_cluster].push_back(i);
        }

        if (rank == 0) {
            std::cout << "Building HNSW for each cluster..." << std::endl;
        }
        hnsw_indices.resize(nlist, nullptr);
        
        #pragma omp parallel for schedule(dynamic)
        for (size_t ci = 0; ci < my_clusters.size(); ++ci) {
            int c = my_clusters[ci];
            size_t cluster_size = clusters_ids[c].size();
            if (cluster_size == 0) continue;
            
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
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Index build and save complete." << std::endl;
        }
    }

    void load(const std::string& index_path) {
        if (rank == 0) {
            quantizer.centroids.resize(nlist * dim);
            std::ifstream in(index_path + "/centroids.bin", std::ios::binary);
            in.read((char*)quantizer.centroids.data(), nlist * dim * sizeof(float));
            in.close();
        }
        
        MPI_Bcast(quantizer.centroids.data(), nlist * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        hnsw_indices.resize(nlist, nullptr);
        
        for (int c : my_clusters) {
            std::string filename = index_path + "/cluster_" + std::to_string(c) + ".bin";
            std::ifstream file_check(filename);
            if (file_check.good()) {
                hnsw_indices[c] = new hnswlib::HierarchicalNSW<float>(ipspace, filename);
            }
        }
        
        std::cout << "Rank " << rank << " loaded " << my_clusters.size() 
                  << " cluster indices." << std::endl;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, int nprobe, int ef_search) 
    {
        // 1. 计算查询到自己负责的簇中心的距离
        std::vector<std::pair<float, int>> local_centroid_dists;
        local_centroid_dists.reserve(my_clusters.size());
        
        for (int c : my_clusters) {
            float dist = computeIPDistance(query, &quantizer.centroids[c * dim], dim);
            local_centroid_dists.push_back({dist, c});
        }
        
        // 2. 收集所有进程的局部距离到 rank 0
        int local_size = local_centroid_dists.size();
        std::vector<int> recv_counts(size);
        MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 修复：计算 float 为单位的偏移量和接收数量
        std::vector<int> displs_float(size, 0);
        std::vector<int> recv_counts_float(size, 0);
        std::vector<float> global_recv_buffer;
        
        if (rank == 0) {
            int total_float = 0;
            for (int i = 0; i < size; ++i) {
                recv_counts_float[i] = recv_counts[i] * 2; // 每个pair展平为2个float
                displs_float[i] = total_float;
                total_float += recv_counts_float[i];
            }
            global_recv_buffer.resize(total_float);
        }
        
        std::vector<float> local_send_buffer(local_size * 2);
        for (int i = 0; i < local_size; ++i) {
            local_send_buffer[i * 2] = local_centroid_dists[i].first;
            local_send_buffer[i * 2 + 1] = static_cast<float>(local_centroid_dists[i].second);
        }
        
        MPI_Gatherv(local_send_buffer.data(), local_size * 2, MPI_FLOAT,
                    global_recv_buffer.data(), recv_counts_float.data(), displs_float.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        // 3. 在 rank 0 上选出全局最近的 nprobe 个簇
        std::vector<int> probe_clusters;
        if (rank == 0) {
            std::vector<std::pair<float, int>> all_centroid_dists;
            for (size_t i = 0; i < global_recv_buffer.size(); i += 2) {
                all_centroid_dists.push_back({global_recv_buffer[i], 
                    static_cast<int>(global_recv_buffer[i+1])});
            }
            
            int actual_nprobe = std::min(nprobe, (int)all_centroid_dists.size());
            std::partial_sort(all_centroid_dists.begin(), 
                             all_centroid_dists.begin() + actual_nprobe,
                             all_centroid_dists.end());
            
            for (int i = 0; i < actual_nprobe; ++i) {
                probe_clusters.push_back(all_centroid_dists[i].second);
            }
        }
        
        // 4. 广播要探测的簇 ID 给所有进程
        int probe_count = probe_clusters.size();
        MPI_Bcast(&probe_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            probe_clusters.resize(probe_count);
        }
        MPI_Bcast(probe_clusters.data(), probe_count, MPI_INT, 0, MPI_COMM_WORLD);
        
        // 5. 每个进程只在自己负责的探测簇中搜索 (利用范围判断替代 std::find 加速)
        std::vector<std::pair<float, uint32_t>> local_results;
        
        for (int cluster_id : probe_clusters) {
            if (cluster_id < start_cluster || cluster_id >= end_cluster) {
                continue; // 不属于本进程负责的簇
            }
            
            if (hnsw_indices[cluster_id] == nullptr) continue;
            
            hnsw_indices[cluster_id]->ef_ = ef_search;
            auto res = hnsw_indices[cluster_id]->searchKnn(query, k);
            
            int sz = res.size();
            std::vector<std::pair<float, uint32_t>> cluster_results(sz);
            while (!res.empty()) {
                cluster_results[--sz] = res.top();
                res.pop();
            }
            
            local_results.insert(local_results.end(), 
                               cluster_results.begin(), cluster_results.end());
        }
        
        // 6. 收集所有进程的局部搜索结果到 rank 0
        int local_result_size = local_results.size();
        std::vector<int> result_recv_counts(size);
        MPI_Gather(&local_result_size, 1, MPI_INT, result_recv_counts.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        
        // 修复：计算 float 为单位的偏移量和接收数量
        std::vector<int> result_displs_float(size, 0);
        std::vector<int> result_recv_counts_float(size, 0);
        std::vector<float> result_recv_buffer;
        
        if (rank == 0) {
            int total_float = 0;
            for (int i = 0; i < size; ++i) {
                result_recv_counts_float[i] = result_recv_counts[i] * 2; // 每个pair展平为2个float
                result_displs_float[i] = total_float;
                total_float += result_recv_counts_float[i];
            }
            result_recv_buffer.resize(total_float);
        }
        
        std::vector<float> result_send_buffer(local_results.size() * 2);
        for (size_t i = 0; i < local_results.size(); ++i) {
            result_send_buffer[i * 2] = local_results[i].first;
            result_send_buffer[i * 2 + 1] = static_cast<float>(local_results[i].second);
        }
        
        MPI_Gatherv(result_send_buffer.data(), local_results.size() * 2, MPI_FLOAT,
                    result_recv_buffer.data(), result_recv_counts_float.data(), result_displs_float.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);
        
        // 7. 在 rank 0 上合并结果并返回 top-k
        std::priority_queue<std::pair<float, uint32_t>> final_res;
        if (rank == 0) {
            std::vector<std::pair<float, uint32_t>> all_results;
            for (size_t i = 0; i < result_recv_buffer.size(); i += 2) {
                all_results.push_back({result_recv_buffer[i], 
                    static_cast<uint32_t>(result_recv_buffer[i+1])});
            }
            
            size_t topk = std::min(k, all_results.size());
            std::partial_sort(all_results.begin(), all_results.begin() + topk, all_results.end());
            
            for (size_t i = 0; i < topk; ++i) {
                final_res.push(all_results[i]);
            }
        }
        
        return final_res;
    }
};

#endif // IVF_HNSW_MPI_H
