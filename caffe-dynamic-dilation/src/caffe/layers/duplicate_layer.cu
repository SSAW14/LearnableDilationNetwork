#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/common.cuh"
#include "caffe/layers/duplicate_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
__global__ void DuplicateForward(const int n, 
  const int channels, const int height, const int width,
  const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    out[index] = in[n * channels + c];
  }
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = top[0]->count();
 
  DuplicateForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, channels_, height_, width_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DuplicateBackward(const int n, 
  const int channels, const int height, const int width,
  const Dtype* in_diff, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    atomicAdd(&out_diff[n * channels + c], in_diff[index]);
  }
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = top[0]->count();

    caffe_gpu_set(bottom[0]->count() , Dtype(0), bottom_diff);

    DuplicateBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, channels_, height_, width_, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
    
    bottom_diff = bottom[0]->mutable_cpu_diff();
    //cout << bottom_diff[0] << endl;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DuplicateLayer);


}  // namespace caffe
