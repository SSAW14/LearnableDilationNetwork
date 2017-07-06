#include <cmath>
#include <vector>
#include <iostream>

#include "caffe/layers/truncation_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {


template <typename Dtype>
__device__ Dtype Max(const Dtype x, const Dtype y) {
  return ((x > y) ? x : y);
}

template <typename Dtype>
__device__ Dtype Min(const Dtype x, const Dtype y) {
  return ((x > y) ? y : x);
}

template <typename Dtype>
__global__ void TruncationForward(const int n, const Dtype* in, const Dtype truncation_min, const Dtype truncation_max, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = Min( Max(in[index], truncation_min), truncation_max);
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  TruncationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, Dtype(truncation_min_), Dtype(truncation_max_), top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void TruncationBackward(const int n, const Dtype* in_diff,
    const Dtype* bottom_data, const Dtype truncation_min, const Dtype truncation_max, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    if (bottom_data[index] > truncation_max) {
      out_diff[index] = Max(in_diff[index], Dtype(0));  
    } else if (bottom_data[index] < truncation_min) {
      out_diff[index] = Min(in_diff[index], Dtype(0));
    } else {
      out_diff[index] = in_diff[index];
    }
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    // NOLINT_NEXT_LINE(whitespace/operators)
    TruncationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, Dtype(truncation_min_), Dtype(truncation_max_), bottom_diff);

    CUDA_POST_KERNEL_CHECK;

  }
}

INSTANTIATE_LAYER_GPU_FUNCS(TruncationLayer);


}  // namespace caffe
