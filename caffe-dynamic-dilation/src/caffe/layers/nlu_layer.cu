#include <cmath>
#include <vector>
#include <iostream>

#include "caffe/layers/nlu_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
__device__ Dtype Abs(const Dtype x) {
  return ((x > Dtype(0)) ? x : (-x));
}

template <typename Dtype>
__device__ Dtype Sign(const Dtype x) {
  return ((x > Dtype(0)) ? 1 : (-1));
}

template <typename Dtype>
__device__ Dtype Max(const Dtype x, const Dtype y) {
  return ((x > y) ? x : y);
}

template <typename Dtype>
__device__ Dtype Min(const Dtype x, const Dtype y) {
  return ((x > y) ? y : x);
}

template <typename Dtype>
__global__ void NLUForward(const int n, const Dtype* in, Dtype ratio, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = Min( Max(in[index] * ratio + Dtype(0.5), Dtype(0)), Dtype(1));
  }
}

template <typename Dtype>
void NLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  NLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, Dtype(ratio_), top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void NLUBackward(const int n, const float alpha, const Dtype* in_diff,
    const Dtype* bottom_data, const Dtype* noise, Dtype ratio, Dtype thre, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ( (Abs(bottom_data[index]) > thre) ? Dtype(0) : ratio)
                    + ( (Abs(bottom_data[index]) > thre) ? ( Sign(bottom_data[index])*alpha*(Abs(bottom_data[index]) - thre) ) : Dtype(0))
                    + noise[index];
  }
}

template <typename Dtype>
void NLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    Dtype* noise_gaussian =
          static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
    Dtype fSTD = sqrt(eta_/pow(1+t_,gamma_));
    caffe_gpu_rng_gaussian(count, Dtype(max_mean_), fSTD, noise_gaussian);

    Dtype dThre = 1.0 / (2 * ratio_);

    // NOLINT_NEXT_LINE(whitespace/operators)
    NLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, this->alpha_, top_diff, bottom_data, noise_gaussian, ratio_, dThre, bottom_diff);

    CUDA_POST_KERNEL_CHECK;

    t_++;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NLULayer);


}  // namespace caffe
