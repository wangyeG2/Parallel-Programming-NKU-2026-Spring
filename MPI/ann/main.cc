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
#include <mpi.h>
//#include "ivf_hnsw_mpi.h"  // 使用MPI版本
//#include "partition_hnsw_mpi.h"
#include "hnsw_on_hnsw_mpi.h"
#include "hnswlib/hnswlib/hnswlib.h"
#include <cfloat>

using namespace hnswlib;

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
    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<" number:"<<n<<" size_per_element:"<<sizeof(T)<<"\n";
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency;
};

int main(int argc, char *argv[]) {
    // 1. 初始化 MPI 环境
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "/anndata/";
    
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;
    
    size_t dimension = vecdim;
    int nlist = 1024;    
    int M = 16;             // HNSW参数
    int ef_construction = 200; // HNSW参数
    std::string index_path = "./files/hnsw_on_hnsw_index"; 
    
    HNSW_on_HNSW_MPI_Index h2_idx(vecdim, nlist, M, ef_construction, rank, size);
    
    // 第一次运行构建索引（构建后注释掉）
    // h2_idx.build_and_save(base, base_number, index_path);
    // MPI_Barrier(MPI_COMM_WORLD);

    // 加载索引
    h2_idx.load(index_path);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) std::cout << "Index loaded from disk." << std::endl;

    int nprobe = 16;       // 搜索时探测的聚类数量
    int ef_search = 100;   // HNSW搜索参数
    
    std::vector<SearchResult> results;
    results.resize(test_number);
    
    // 3. 查询测试
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);
        
        auto res = h2_idx.search(test_query + i * vecdim, k, nprobe, ef_search);
        
        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - 
                       (val.tv_sec * Converter + val.tv_usec);
        
        if (rank == 0) {
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
    }
    
    // 4. 只有 rank 0 负责输出最终性能数据
    if (rank == 0) {
        float avg_recall = 0, avg_latency = 0;
        for(int i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }
        std::cout << "======= MPI Performance Test Results =======" << std::endl;
        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    }

    // 5. 清理资源并结束 MPI
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    MPI_Finalize();
    return 0;
}
