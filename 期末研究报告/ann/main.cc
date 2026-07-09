#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>

// ==================== 包含 HNSW 库 ====================
#include "hnswlib/hnswlib/hnswlib.h"

// ==================== 包含所有算法头文件 ====================
#include "flat_scan.h"
#include "flat_simd_omp.h"
#include "flat_simd2.h"
#include "sq_simd.h"
#include "pq_simd2.h"

#if defined(USE_IVF_PQ)
#include "ivf-pq-simd.h"
#endif
#if defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT)
#include <mpi.h>
#if defined(USE_IVF_MPI_MT)
#include "ivf-simd-mpi-mt.h"
#else
#include "ivf-simd-mpi.h"
#endif
#endif
#if defined(USE_HNSW)
#include "hnswlib/hnswlib/hnswlib.h"
#endif
#if defined(USE_IVF_HNSW)
#include "ivf_hnsw.h"
#endif
#if defined(USE_IVF_HNSW_MPI)
#include "ivf_hnsw_mpi.h"
#endif
#if defined(USE_PARTITION_HNSW_MPI)
#include "partition_hnsw_mpi.h"
#endif
#if defined(USE_HNSW_ON_HNSW_MPI)
#include "hnsw_on_hnsw_mpi.h"
#endif

// ==================== 默认参数宏定义 (供脚本通过 -D 传入) ====================
#ifndef ALG_THREADS
#define ALG_THREADS 8
#endif
#ifndef ALG_P
#define ALG_P 100
#endif
#ifndef ALG_NPROBE
#define ALG_NPROBE 10
#endif
#ifndef ALG_NLIST
#define ALG_NLIST 100
#endif
#ifndef ALG_HNSW_M
#define ALG_HNSW_M 16
#endif
#ifndef ALG_HNSW_EF_CONSTRUCTION
#define ALG_HNSW_EF_CONSTRUCTION 200
#endif
#ifndef ALG_HNSW_EF_SEARCH
#define ALG_HNSW_EF_SEARCH 100
#endif

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d) {
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim) {
    // 保留原始的 build_index 函数，避免编译问题
}

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    // 只测试前2000条查询
    test_number = 2000;
    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // ==================== MPI 初始化 ====================
#if defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT) || defined(USE_IVF_HNSW_MPI) || defined(USE_PARTITION_HNSW_MPI) || defined(USE_HNSW_ON_HNSW_MPI)
    int mpi_rank = 0, mpi_size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

    // ==================== 索引构建区 (根据编译宏) ====================
#if defined(USE_SQ)
    SQIndex sq_idx;
    build_sq_index(base, base_number, vecdim, sq_idx);
#endif
#if defined(USE_PQ)
    PQIndex pq_idx;
    build_pq_index(base, base_number, vecdim, pq_idx);
#endif
#if defined(USE_IVF_PQ)
    IVFPQIndex ivf_pq_idx;
    build_ivf_pq_index(base, base_number, vecdim, ivf_pq_idx, ALG_NLIST);
#endif
#if defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT)
    IVFIndex ivf_mpi_idx;
    build_ivf_index_mpi(base, base_number, vecdim, ivf_mpi_idx, ALG_NLIST, mpi_rank, mpi_size);
#endif
#if defined(USE_HNSW)
    hnswlib::L2Space l2space(vecdim);
    hnswlib::HierarchicalNSW<float> hnsw_index(&l2space, base_number, ALG_HNSW_M, ALG_HNSW_EF_CONSTRUCTION);
    for (size_t i = 0; i < base_number; ++i) {
        hnsw_index.addPoint(base + i * vecdim, i);
    }
#endif
#if defined(USE_IVF_HNSW)
    IVF_HNSW_Index ivf_hnsw_idx(vecdim, ALG_NLIST, ALG_HNSW_M, ALG_HNSW_EF_CONSTRUCTION);
    //ivf_hnsw_idx.build_and_save(base, base_number, "ivf_hnsw_index");
    ivf_hnsw_idx.load("ivf_hnsw_index");
#endif
#if defined(USE_IVF_HNSW_MPI)
    IVF_HNSW_MPI_Index ivf_hnsw_mpi_idx(vecdim, ALG_NLIST, ALG_HNSW_M, ALG_HNSW_EF_CONSTRUCTION, mpi_rank, mpi_size);
    //ivf_hnsw_mpi_idx.build_and_save(base, base_number, "ivf_hnsw_mpi_index");
    ivf_hnsw_mpi_idx.load("ivf_hnsw_mpi_index");
#endif
#if defined(USE_PARTITION_HNSW_MPI)
    Partition_HNSW_MPI_Index partition_hnsw_mpi_idx(vecdim, ALG_HNSW_M, ALG_HNSW_EF_CONSTRUCTION, mpi_rank, mpi_size);
    //partition_hnsw_mpi_idx.build_and_save(base, base_number, "partition_hnsw_mpi_index");
    partition_hnsw_mpi_idx.load("partition_hnsw_mpi_index", base_number);
#endif
#if defined(USE_HNSW_ON_HNSW_MPI)
    HNSW_on_HNSW_MPI_Index hnsw_on_hnsw_mpi_idx(vecdim, ALG_NLIST, ALG_HNSW_M, ALG_HNSW_EF_CONSTRUCTION, mpi_rank, mpi_size);
    //hnsw_on_hnsw_mpi_idx.build_and_save(base, base_number, "hnsw_on_hnsw_mpi_index");
    hnsw_on_hnsw_mpi_idx.load("hnsw_on_hnsw_mpi_index");
#endif

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 该文件已有代码中你只能修改该函数的调用方式
        // 可以任意修改函数名，函数参数或者改为调用成员函数，但是不能修改函数返回值。
        std::priority_queue<std::pair<float, uint32_t>> res;

#if defined(USE_FLAT)
        res = flat_search(base, test_query + i * vecdim, base_number, vecdim, k);
#elif defined(USE_FLAT_SIMD_OMP)
        res = flat_simd_omp_search(base, test_query + i * vecdim, base_number, vecdim, k, ALG_THREADS, k);
#elif defined(USE_FLAT_SIMD_PTHREAD)
        res = pthread_simd_flat_search(base, test_query + i * vecdim, base_number, vecdim, k, ALG_THREADS);
#elif defined(USE_SQ)
        res = sq_search(sq_idx, base, test_query + i * vecdim, base_number, vecdim, k, ALG_P);
#elif defined(USE_PQ)
        res = pq_search_pthread(pq_idx, base, test_query + i * vecdim, base_number, vecdim, k, ALG_P, ALG_THREADS);
#elif defined(USE_IVF_PQ)
        res = ivf_pq_simd_search(ivf_pq_idx, base, test_query + i * vecdim, base_number, vecdim, k, ALG_NPROBE, ALG_P);
#elif defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT)
        res = ivf_simd_mpi_search(ivf_mpi_idx, base, test_query + i * vecdim, base_number, vecdim, k, ALG_NPROBE, mpi_rank, mpi_size);
#elif defined(USE_HNSW)
        hnsw_index.ef_ = ALG_HNSW_EF_SEARCH;
        res = hnsw_index.searchKnn(test_query + i * vecdim, k);
#elif defined(USE_IVF_HNSW)
        res = ivf_hnsw_idx.search(test_query + i * vecdim, k, ALG_NPROBE, ALG_HNSW_EF_SEARCH);
#elif defined(USE_IVF_HNSW_MPI)
        res = ivf_hnsw_mpi_idx.search(test_query + i * vecdim, k, ALG_NPROBE, ALG_HNSW_EF_SEARCH);
#elif defined(USE_PARTITION_HNSW_MPI)
        res = partition_hnsw_mpi_idx.search(test_query + i * vecdim, k, ALG_HNSW_EF_SEARCH);
#elif defined(USE_HNSW_ON_HNSW_MPI)
        res = hnsw_on_hnsw_mpi_idx.search(test_query + i * vecdim, k, ALG_NPROBE, ALG_HNSW_EF_SEARCH);
#else
        // 默认兜底
        res = flat_search(base, test_query + i * vecdim, base_number, vecdim, k);
#endif

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }
        size_t acc = 0;
        while (res.size()) {
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc/k;
        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

#if defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT) || defined(USE_IVF_HNSW_MPI) || defined(USE_PARTITION_HNSW_MPI) || defined(USE_HNSW_ON_HNSW_MPI)
    if (mpi_rank == 0) {
        std::cout << "average recall: "<<avg_recall / test_number<<"\n";
        std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    }
#else
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
#endif

#if defined(USE_IVF_MPI) || defined(USE_IVF_MPI_MT) || defined(USE_IVF_HNSW_MPI) || defined(USE_PARTITION_HNSW_MPI) || defined(USE_HNSW_ON_HNSW_MPI)
    MPI_Finalize();
#endif

    return 0;
}
