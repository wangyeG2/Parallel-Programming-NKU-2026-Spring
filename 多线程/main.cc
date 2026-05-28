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
#include "ivf_hnsw.h"
#include "flat_simd_omp.h"
//#include "sq_simd.h"
//#include "pq_simd.h" 注意，由于函数调用接口一致，pq_simd.h 和 pq_simd2.h 只能保留一个，你可以根据需要选择使用哪个版本的 PQ 实现，另一个版本请删除或者注释掉。
//#include "pq_simd2.h"
#include <cfloat>
// 可以自行添加需要的头文件

using namespace hnswlib;
hnswlib::InnerProductSpace *ipspace = nullptr;
hnswlib::HierarchicalNSW<float> *appr_alg = nullptr;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
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
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

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


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 只测试前2000条查询
    test_number = 2000;
    const size_t k = 10;
    std::vector<int> thread_list = {1, 2, 4, 8}; // 根据服务器CPU核心数调整
    std::vector<omp_sched_t> sched_list = {omp_sched_static, omp_sched_dynamic, omp_sched_guided};
    std::vector<int> chunk_list = {0, 1, 4, 16, 64, 256, 1024}; // 0表示OpenMP自动选择
    std::vector<size_t> local_p_list = {10, 20, 50, 100}; // 局部Top-P设置
    std::vector<SearchResult> results;
    results.resize(test_number);
    std::cout << "Threads,Schedule,Chunk,LocalP,AvgRecall,AvgLatency(us)" << std::endl;
    // 如果你需要保存索引，可以在这里添加你需要的函数，你可以将下面的注释删除来查看pbs是否将build.index返回到你的files目录中
    // 要保存的目录必须是files/
    // 每个人的目录空间有限，不需要的索引请及时删除，避免占空间太大
    // 不建议在正式测试查询时同时构建索引，否则性能波动会较大
    // 下面是一个构建hnsw索引的示例
    

    
    for (int num_threads : thread_list) {
        // 遍历调度策略
        for (omp_sched_t sched : sched_list) {
            // 遍历 Chunk Size
            for (int chunk : chunk_list) {
                // 遍历局部 Top-P
                for (size_t local_p : local_p_list) {
                    
                    // 设置运行时调度策略
                    omp_set_schedule(sched, chunk);

                    std::vector<SearchResult> results(test_number);
                    for(int i = 0; i < test_number; ++i) {
                        struct timeval val, newVal;
                        gettimeofday(&val, NULL);

                        // 调用修改后的函数，传入 local_p
                        auto res = flat_simd_omp_search(base, test_query + i * vecdim, 
                                                        base_number, vecdim, k, num_threads, local_p);

                        gettimeofday(&newVal, NULL);
                        int64_t diff = (newVal.tv_sec * 1000000 + newVal.tv_usec) - (val.tv_sec * 1000000 + val.tv_usec);

                        std::set<uint32_t> gtset;
                        for(int j = 0; j < k; ++j) gtset.insert(test_gt[j + i*test_gt_d]);
                        
                        size_t acc = 0;
                        while (!res.empty()) {
                            if(gtset.find(res.top().second) != gtset.end()) ++acc;
                            res.pop();
                        }
                        results[i] = {(float)acc/k, diff};
                    }

                    // 计算平均值
                    float avg_recall = 0, avg_latency = 0;
                    for(int i = 0; i < test_number; ++i) {
                        avg_recall += results[i].recall;
                        avg_latency += results[i].latency;
                    }
                    avg_recall /= test_number;
                    avg_latency /= test_number;

                    // 输出当前参数组合的结果
                    std::string sched_name;
                    if(sched == omp_sched_static) sched_name = "static";
                    else if(sched == omp_sched_dynamic) sched_name = "dynamic";
                    else sched_name = "guided";

                    std::cout << num_threads << "," 
                              << sched_name << "," 
                              << chunk << "," 
                              << local_p << "," 
                              << avg_recall << "," 
                              << avg_latency << std::endl;
                }
            }
        }
    }
    // ... delete 部分保持不变 ...
}