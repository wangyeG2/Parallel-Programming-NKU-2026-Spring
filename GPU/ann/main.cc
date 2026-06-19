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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// #include "flat_scan_simd.h" // 替换为我们的GPU版本
//#include "gpu_search.cuh"      // 引入GPU搜索头文件
//#include "gpu_search_tiling.cuh"
#include "gpu_search_ivf.cuh"
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
    return data;
}

struct SearchResult {
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim) {
    const int efConstruction = 150;
    const int M = 16;
    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);
    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }
    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

int main(int argc, char *argv[]) {
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    std::string data_path = "./";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);

    test_number = 2000;
    const size_t k = 10;
    std::vector<SearchResult> results;
    results.resize(test_number);

    // ================= GPU初始化与数据预加载 =================
    // 提前将base数据上传至GPU显存，避免每次查询重复传输
    init_ivf_gpu(base, base_number, vecdim, 100);

    // 设置Batch Size，体现查询间并行
    const size_t batch_size = 100; 
    std::vector<SearchResult>().swap(results); results.resize(test_number);

    // ================= 批量查询测试代码 =================
    for(size_t i = 0; i < test_number; i += batch_size) {
        size_t current_batch = std::min(batch_size, test_number - i);
        
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        // 调用GPU批量搜索函数，返回current_batch个优先队列
        auto res_batch = grouped_gpu_search_batch(test_query + i * vecdim, base_number, vecdim, k, current_batch, 100);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        // 计算该batch的总耗时，并均摊到每个查询上计算平均延迟
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
        int64_t avg_diff = diff / current_batch;

        for(size_t b = 0; b < current_batch; ++b) {
            auto res = res_batch[b];
            std::set<uint32_t> gtset;
            for(int j = 0; j < k; ++j){
                int t = test_gt[j + (i+b)*test_gt_d];
                gtset.insert(t);
            }
            size_t acc = 0;
            while (res.size()) {
                int x = res.top().second;
                if(gtset.find(x) != gtset.end()){ ++acc; }
                res.pop();
            }
            float recall = (float)acc / k;
            results[i+b] = {recall, avg_diff};
        }
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "average recall: "<< avg_recall / test_number <<"\n";
    std::cout << "average latency (us): "<< avg_latency / test_number <<"\n";
    
    free_ivf_gpu();
    return 0;
}
