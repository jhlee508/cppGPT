#include <cstring>
#include <cstdlib>

#include "tensor.h"
#include "util.h"

using namespace std;

Tensor::Tensor(const vector<int> &shape_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { 
        shape[i] = shape_[i]; 
    }
    int N_ = num_elem();
    buf = (float *)calloc(N_, sizeof(float));
}

Tensor::Tensor(const vector<int> &shape_, float *buf_) {
    ndim = shape_.size();
    for (size_t i = 0; i < ndim; i++) { 
        shape[i] = shape_[i]; 
    }
    int N_ = num_elem();
    buf = (float *)malloc(N_ * sizeof(float));
    memcpy(buf, buf_, N_ * sizeof(float));
}

Tensor::~Tensor() { 
    if (buf != nullptr) free(buf); 
}

int Tensor::num_elem() {
    int size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= shape[i];
    }
    return size;
}