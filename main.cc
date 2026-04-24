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
// 可以自行添加需要的头文件
#include <arm_neon.h>
#include <cmath>
using namespace hnswlib;

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

// NEON 优化的内积距离计算 (1 - IP)
inline float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim) {
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    
    size_t i = 0;
    // 每次循环处理 8 个 float (循环展开)
    for (; i + 7 < vecdim; i += 8) {
        // 加载底库向量数据
        float32x4_t v1_0 = vld1q_f32(b1 + i);
        float32x4_t v1_1 = vld1q_f32(b1 + i + 4);
        
        // 加载查询向量数据
        float32x4_t v2_0 = vld1q_f32(b2 + i);
        float32x4_t v2_1 = vld1q_f32(b2 + i + 4);
        
        // 乘加运算：sum_vec = sum_vec + v1 * v2
        sum_vec0 = vmlaq_f32(sum_vec0, v1_0, v2_0);
        sum_vec1 = vmlaq_f32(sum_vec1, v1_1, v2_1);
    }
    
    // 将两组累加结果合并
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
    
    // AArch64 特有的水平求和指令，直接把 float32x4_t 里的 4 个 float 加起来
    float sum = vaddvq_f32(sum_vec0); 
    
    // 处理尾部不能被 8 整除的剩余元素
    for (; i < vecdim; ++i) {
        sum += b1[i] * b2[i];
    }
    
    // 返回距离
    return 1.0f - sum;
}

// 全新的搜索函数
std::priority_queue<std::pair<float, uint32_t> > flat_search_neon(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for(size_t i = 0; i < base_number; ++i) {
        // 调用我们写的 NEON 距离计算函数
        float dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);

        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
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

// --- 1. 量化函数 ---
inline void quantize_to_int8(const float* src, int8_t* dst, size_t size, float scale) {
    for (size_t i = 0; i < size; ++i) {
        float val = src[i] / scale;
        if (val > 127.0f) val = 127.0f;
        if (val < -127.0f) val = -127.0f;
        dst[i] = (int8_t)std::round(val);
    }
}

// --- 2. NEON 优化的 int8 粗排内积计算 (16路并发) ---
inline int32_t InnerProductSIMDNeon_Int8(const int8_t* b1, const int8_t* b2, size_t vecdim) {
    int32x4_t sum_vec = vdupq_n_s32(0);
    size_t i = 0;
    
    // 一次处理 16 个 int8 数据
    for (; i + 15 < vecdim; i += 16) {
        int8x16_t v1 = vld1q_s8(b1 + i);
        int8x16_t v2 = vld1q_s8(b2 + i);

        // 把 16 个 int8 拆成低 8 位和高 8 位进行长乘法，结果变成 16 位
        int16x8_t p1 = vmull_s8(vget_low_s8(v1), vget_low_s8(v2));
        int16x8_t p2 = vmull_s8(vget_high_s8(v1), vget_high_s8(v2));

        // 将相邻的 16 位结果相加，并累加到 32 位寄存器中
        sum_vec = vpadalq_s16(sum_vec, p1);
        sum_vec = vpadalq_s16(sum_vec, p2);
    }
    
    // 水平求和，把 4 个 32 位累加结果加成一个标量
    int32_t sum = vaddvq_s32(sum_vec);
    
    // 尾部处理
    for (; i < vecdim; ++i) {
        sum += (int32_t)b1[i] * (int32_t)b2[i];
    }
    return sum;
}

// --- 3. 两阶段检索：SQ 搜索函数 ---
std::priority_queue<std::pair<float, uint32_t> > sq_search_neon(
    const int8_t* q_base, const float* base, const float* query, 
    size_t base_number, size_t vecdim, size_t k, float scale) 
{
    // [第一阶段]：量化当前的 Query
    std::vector<int8_t> q_query(vecdim);
    quantize_to_int8(query, q_query.data(), vecdim, scale);

    // [第二阶段]：粗排 (Coarse Search) - 选出 Top P
    size_t P = 20; // 候选集大小，可调节
    // 使用最小堆来保留内积最大的 P 个候选人
    std::priority_queue<std::pair<int32_t, uint32_t>, std::vector<std::pair<int32_t, uint32_t>>, std::greater<std::pair<int32_t, uint32_t>>> coarse_q; 

    for (size_t i = 0; i < base_number; ++i) {
        int32_t ip_int = InnerProductSIMDNeon_Int8(q_base + i * vecdim, q_query.data(), vecdim);
        if (coarse_q.size() < P) {
            coarse_q.push({ip_int, i});
        } else if (ip_int > coarse_q.top().first) {
            coarse_q.pop();
            coarse_q.push({ip_int, i});
        }
    }

    // [第三阶段]：精排 (Rerank)
    std::priority_queue<std::pair<float, uint32_t> > fine_q; // 默认最大堆，保留距离最小的 k 个
    while (!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();

        // 拿粗排筛出来的索引，回到 float 空间用之前的 NEON 函数算精确距离
        float dis = InnerProductSIMDNeon(base + idx * vecdim, query, vecdim);

        if (fine_q.size() < k) {
            fine_q.push({dis, idx});
        } else if (dis < fine_q.top().first) {
            fine_q.pop();
            fine_q.push({dis, idx});
        }
    }
    return fine_q;
}
// --- [对比实验用] 纯串行的 int8 内积计算 ---
inline int32_t InnerProductSerial_Int8(const int8_t* b1, const int8_t* b2, size_t vecdim) {
    int32_t sum = 0;
    // 故意写成最简单的循环，让编译器去尝试自动向量化 (Auto-Vectorization)
    for (size_t i = 0; i < vecdim; ++i) {
        sum += (int32_t)b1[i] * (int32_t)b2[i];
    }
    return sum;
}

// --- [对比实验用] 纯串行的 float 内积计算 ---
inline float InnerProductSerial_Float(const float* b1, const float* b2, size_t vecdim) {
    float sum = 0.0f;
    for (size_t i = 0; i < vecdim; ++i) {
        sum += b1[i] * b2[i];
    }
    return 1.0f - sum;
}

// --- [对比实验用] 纯串行 SQ 搜索函数 ---
std::priority_queue<std::pair<float, uint32_t> > sq_search_serial(
    const int8_t* q_base, const float* base, const float* query, 
    size_t base_number, size_t vecdim, size_t k, float scale) 
{
    std::vector<int8_t> q_query(vecdim);
    quantize_to_int8(query, q_query.data(), vecdim, scale);

    size_t P = 20; // 保持和你刚才最后一次测试一致
    std::priority_queue<std::pair<int32_t, uint32_t>, std::vector<std::pair<int32_t, uint32_t>>, std::greater<std::pair<int32_t, uint32_t>>> coarse_q; 

    for (size_t i = 0; i < base_number; ++i) {
        // 调用串行 int8 计算
        int32_t ip_int = InnerProductSerial_Int8(q_base + i * vecdim, q_query.data(), vecdim);
        if (coarse_q.size() < P) {
            coarse_q.push({ip_int, i});
        } else if (ip_int > coarse_q.top().first) {
            coarse_q.pop();
            coarse_q.push({ip_int, i});
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > fine_q; 
    while (!coarse_q.empty()) {
        uint32_t idx = coarse_q.top().second;
        coarse_q.pop();

        // 调用串行 float 计算
        float dis = InnerProductSerial_Float(base + idx * vecdim, query, vecdim);

        if (fine_q.size() < k) {
            fine_q.push({dis, idx});
        } else if (dis < fine_q.top().first) {
            fine_q.pop();
            fine_q.push({dis, idx});
        }
    }
    return fine_q;
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

    // ================== 新增：预处理量化底库 ==================
    float max_abs_val = 0.523538f; // 根据刚才跑出来的值设定
    float scale = max_abs_val / 127.0f;
    
    std::cout << "Quantizing base vectors..." << std::endl;
    std::vector<int8_t> q_base(base_number * vecdim);
    
    // 使用 OpenMP 多线程加速量化过程
    #pragma omp parallel for
    for (size_t i = 0; i < base_number; ++i) {
        quantize_to_int8(base + i * vecdim, q_base.data() + i * vecdim, vecdim, scale);
    }
    std::cout << "Quantization done." << std::endl;
    // ==========================================================

    std::vector<SearchResult> results;
    results.resize(test_number);

    // 查询测试代码
    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 调用我们新鲜出炉的 SQ-SIMD 两阶段搜索函数
        // 将原来的 auto res = sq_search_neon(...) 注释掉，换成下面这行：
        auto res = sq_search_neon(q_base.data(), base, test_query + i*vecdim, base_number, vecdim, k, scale);

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

    // 浮点误差可能导致一些精确算法平均recall不是1
    std::cout << "average recall: "<<avg_recall / test_number<<"\n";
    std::cout << "average latency (us): "<<avg_latency / test_number<<"\n";
    return 0;
}