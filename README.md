# ANN SIMD Experiment

本项目为并行程序设计 SIMD 编程实验的 ANN 选题实现。

## 内容
- Flat 串行基线
- Flat-NEON 向量化实现
- SQ-Serial 两阶段检索
- SQ-NEON 两阶段检索
- NoVec / AutoVec / Handwritten NEON 对比实验

## 文件说明
- main.cc: 主程序与 SIMD 实现
- flat_scan.h: Flat 串行基线
- hnswlib/: 依赖库
- qsub.sh / test.sh: 运行脚本

## 编译运行
g++ -O3 -std=c++17 -march=native -fopenmp main.cc -o ann_run
./ann_run
