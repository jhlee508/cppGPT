#pragma once

#include <cstdio>
#include <vector>

using namespace std;

struct Tensor {
    size_t ndim = 0;
    int shape[4];
    float *buf = nullptr;

    Tensor(const vector<int> &shape_);
    Tensor(const vector<int> &shape_, float *buf_);
    ~Tensor();

    int num_elem();
};