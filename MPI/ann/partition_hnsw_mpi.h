#ifndef PARTITION_HNSW_MPI_H
#define PARTITION_HNSW_MPI_H

#include "hnswlib/hnswlib/hnswlib.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <sys/stat.h>
#include <mpi.h>

class Partition_HNSW_MPI_Index {
public:
    size_t dim;
    int M;
    int ef_construction;
    hnswlib::InnerProductSpace* ipspace;
    hnswlib::HierarchicalNSW<float>* hnsw_index; // 每个进程只维护一个HNSW
    int rank;
    int size;
    size_t local_n;       // 本进程负责的数据量
    size_t local_offset;  // 本进程数据在全局base中的起始偏移量

    Partition_HNSW_MPI_Index(size_t dimension, int m, int ef_constr, 
                             int mpi_rank, int mpi_size) 
        : dim(dimension), M(m), ef_construction(ef_constr), 
          rank(mpi_rank), size(mpi_size), hnsw_index(nullptr) {
        ipspace = new hnswlib::InnerProductSpace(dim);
    }

    ~Partition_HNSW_MPI_Index() {
        if (hnsw_index) delete hnsw_index;
        delete ipspace;
    }

    void create_directory(const std::string& path) {
        if (rank == 0) mkdir(path.c_str(), 0777);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void build_and_save(float* base, size_t n, const std::string& index_path) {
        create_directory(index_path);

        // 1. 计算每个进程负责的数据范围 (简单均分)
        size_t base_n = n / size;
        size_t remainder = n % size;
        local_n = base_n + (rank < remainder ? 1 : 0);
        local_offset = rank * base_n + std::min(rank, (int)remainder);

        std::cout << "Rank " << rank << " building HNSW for data [" 
                  << local_offset << ", " << local_offset + local_n << ")..." << std::endl;

        // 2. 拷贝本进程负责的数据
        std::vector<float> local_data(local_n * dim);
        memcpy(local_data.data(), base + local_offset * dim, local_n * dim * sizeof(float));

        // 3. 构建本地 HNSW (注意：ID需要加上偏移量，使其对应全局ID)
        hnsw_index = new hnswlib::HierarchicalNSW<float>(ipspace, local_n, M, ef_construction);
        
        if (local_n > 0) {
            hnsw_index->addPoint(local_data.data(), local_offset); // 第一个点
            #pragma omp parallel for schedule(dynamic)
            for (size_t i = 1; i < local_n; ++i) {
                hnsw_index->addPoint(&local_data[i * dim], local_offset + i);
            }
        }

        // 4. 保存索引
        hnsw_index->saveIndex(index_path + "/partition_" + std::to_string(rank) + ".bin");
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) std::cout << "Partition HNSW build and save complete." << std::endl;
    }

    void load(const std::string& index_path, size_t n) {
        size_t base_n = n / size;
        size_t remainder = n % size;
        local_n = base_n + (rank < remainder ? 1 : 0);
        local_offset = rank * base_n + std::min(rank, (int)remainder);

        std::string filename = index_path + "/partition_" + std::to_string(rank) + ".bin";
        std::ifstream file_check(filename);
        if (file_check.good()) {
            hnsw_index = new hnswlib::HierarchicalNSW<float>(ipspace, filename);
        } else {
            std::cerr << "Rank " << rank << " failed to load index file!" << std::endl;
        }
        std::cout << "Rank " << rank << " loaded partition index." << std::endl;
    }

    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, int ef_search) 
    {
        std::vector<std::pair<float, uint32_t>> local_results;

        // 1. 每个进程在自己的 HNSW 中搜索 top-k
        if (hnsw_index != nullptr && local_n > 0) {
            hnsw_index->ef_ = ef_search;
            auto res = hnsw_index->searchKnn(query, k);
            
            int sz = res.size();
            local_results.resize(sz);
            while (!res.empty()) {
                local_results[--sz] = res.top();
                res.pop();
            }
        }

        // 2. 收集所有进程的结果到 Rank 0
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
                    result_recv_buffer.data(), result_recv_counts_float.data(), result_displs_float.data(), MPI_FLOAT,
                    0, MPI_COMM_WORLD);

        // 3. 在 Rank 0 上合并全局结果
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

#endif // PARTITION_HNSW_MPI_H
