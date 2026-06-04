#ifndef HNSW_ON_HNSW_MPI_H
#define HNSW_ON_HNSW_MPI_H

#include "hnswlib/hnswlib/hnswlib.h"
#include "kmeans.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <sys/stat.h>
#include <mpi.h>

class HNSW_on_HNSW_MPI_Index {
public:
    size_t dim;
    int nlist;
    int M;
    int ef_construction;
    KMeans quantizer;
    hnswlib::InnerProductSpace* ipspace;
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_indices;

    // 新增：顶层粗筛 HNSW
    hnswlib::HierarchicalNSW<float>* coarse_hnsw;
    int coarse_M;
    int coarse_ef_construction;

    int rank;
    int size;
    std::vector<int> my_clusters;
    int start_cluster;
    int end_cluster;

    HNSW_on_HNSW_MPI_Index(size_t dimension, int num_clusters, int m, int ef_constr, int mpi_rank, int mpi_size)
        : dim(dimension), nlist(num_clusters), M(m), ef_construction(ef_constr),
          quantizer(dimension, num_clusters), rank(mpi_rank), size(mpi_size),
          coarse_hnsw(nullptr), coarse_M(32), coarse_ef_construction(200) { // 粗筛图参数可调
        ipspace = new hnswlib::InnerProductSpace(dim);
        int clusters_per_rank = nlist / size;
        int remainder = nlist % size;
        start_cluster = rank * clusters_per_rank + std::min(rank, remainder);
        end_cluster = start_cluster + clusters_per_rank + (rank < remainder ? 1 : 0);
        for (int c = start_cluster; c < end_cluster; ++c) {
            my_clusters.push_back(c);
        }
    }

    ~HNSW_on_HNSW_MPI_Index() {
        for (auto* idx : hnsw_indices) {
            if(idx) delete idx;
        }
        if (coarse_hnsw) delete coarse_hnsw;
        delete ipspace;
    }

    void create_directory(const std::string& path) {
        if (rank == 0) mkdir(path.c_str(), 0777);
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

            // 新增：在 Rank 0 上构建顶层粗筛 HNSW
            std::cout << "Building coarse HNSW on centroids..." << std::endl;
            coarse_hnsw = new hnswlib::HierarchicalNSW<float>(ipspace, nlist, coarse_M, coarse_ef_construction);
            for (int c = 0; c < nlist; ++c) {
                coarse_hnsw->addPoint(&quantizer.centroids[c * dim], c);
            }
            coarse_hnsw->saveIndex(index_path + "/coarse_hnsw.bin");
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
                base + i * dim,
                base + (i + 1) * dim
            );
            clusters_ids[best_cluster].push_back(i);
        }

        if (rank == 0) std::cout << "Building HNSW for each cluster..." << std::endl;
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
        if (rank == 0) std::cout << "Index build and save complete." << std::endl;
    }

    void load(const std::string& index_path) {
        if (rank == 0) {
            quantizer.centroids.resize(nlist * dim);
            std::ifstream in(index_path + "/centroids.bin", std::ios::binary);
            in.read((char*)quantizer.centroids.data(), nlist * dim * sizeof(float));
            in.close();

            // 修改：只在 Rank 0 上构建粗筛 HNSW
            if (coarse_hnsw) delete coarse_hnsw;
            coarse_hnsw = new hnswlib::HierarchicalNSW<float>(ipspace, nlist, coarse_M, coarse_ef_construction);
            for (int c = 0; c < nlist; ++c) {
                coarse_hnsw->addPoint(&quantizer.centroids[c * dim], c);
            }
            std::cout << "Rank 0 built coarse HNSW in memory." << std::endl;
        }
        // 广播质心给其他进程，但不需要广播 coarse_hnsw
        MPI_Bcast(quantizer.centroids.data(), nlist * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

        hnsw_indices.resize(nlist, nullptr);
        for (int c : my_clusters) {
            std::string filename = index_path + "/cluster_" + std::to_string(c) + ".bin";
            std::ifstream file_check(filename);
            if (file_check.good()) {
                hnsw_indices[c] = new hnswlib::HierarchicalNSW<float>(ipspace, filename);
            }
        }
        std::cout << "Rank " << rank << " loaded " << my_clusters.size() << " cluster indices." << std::endl;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, int nprobe, int ef_search) {
        // ==========================================
        // 1. 粗筛阶段：修正为在 Rank 0 搜索后广播簇 ID
        // ==========================================
        std::vector<int> probe_clusters;
        if (rank == 0) {
            coarse_hnsw->ef_ = std::max(nprobe * 2, 64); // 设置粗搜 ef
            auto coarse_res = coarse_hnsw->searchKnn(query, nprobe);
            while (!coarse_res.empty()) {
                probe_clusters.push_back(coarse_res.top().second);
                coarse_res.pop();
            }
        }

        // 广播探测簇数量和ID
        int probe_count = probe_clusters.size();
        MPI_Bcast(&probe_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            probe_clusters.resize(probe_count);
        }
        MPI_Bcast(probe_clusters.data(), probe_count, MPI_INT, 0, MPI_COMM_WORLD);

        // ==========================================
        // 2. 细查阶段：每个进程在自己负责的探测簇中搜索
        // ==========================================
        std::vector<std::pair<float, uint32_t>> local_results;
        for (int cluster_id : probe_clusters) {
            if (cluster_id < start_cluster || cluster_id >= end_cluster) continue;
            if (hnsw_indices[cluster_id] == nullptr) continue;
            hnsw_indices[cluster_id]->ef_ = ef_search;
            auto res = hnsw_indices[cluster_id]->searchKnn(query, k);
            int sz = res.size();
            std::vector<std::pair<float, uint32_t>> cluster_results(sz);
            while (!res.empty()) {
                cluster_results[--sz] = res.top();
                res.pop();
            }
            local_results.insert(local_results.end(), cluster_results.begin(), cluster_results.end());
        }

        // ==========================================
        // 3. 合并阶段：收集局部结果到 Rank 0
        // ==========================================
        int local_result_size = local_results.size();
        std::vector<int> result_recv_counts(size);
        MPI_Gather(&local_result_size, 1, MPI_INT, result_recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> result_displs_float(size, 0);
        std::vector<int> result_recv_counts_float(size, 0);
        std::vector<float> result_recv_buffer;
        if (rank == 0) {
            int total_float = 0;
            for (int i = 0; i < size; ++i) {
                result_recv_counts_float[i] = result_recv_counts[i] * 2;
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
                    result_recv_buffer.data(), result_recv_counts_float.data(), result_displs_float.data(),
                    MPI_FLOAT, 0, MPI_COMM_WORLD);

        // ==========================================
        // 4. Rank 0 全局 Top-K 合并
        // ==========================================
        std::priority_queue<std::pair<float, uint32_t>> final_res;
        if (rank == 0) {
            std::vector<std::pair<float, uint32_t>> all_results;
            for (size_t i = 0; i < result_recv_buffer.size(); i += 2) {
                all_results.push_back({result_recv_buffer[i], static_cast<uint32_t>(result_recv_buffer[i+1])});
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

#endif // HNSW_ON_HNSW_MPI_H
