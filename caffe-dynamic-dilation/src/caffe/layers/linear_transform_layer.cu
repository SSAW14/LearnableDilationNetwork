#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/linear_transform_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LinearTransformForward(const int n, const Dtype* in, Dtype* out,
    Dtype scale, Dtype bias) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * scale + bias;
  }
}

template <typename Dtype>
void LinearTransformLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
 
  LinearTransformForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, Dtype(scale_), Dtype(bias_));
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void LinearTransformBackward(const int n, const Dtype* in_diff,
    Dtype* out_diff, Dtype scale) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale;
  }
}

template <typename Dtype>
void LinearTransformLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();

    LinearTransformBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, Dtype(scale_));
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(LinearTransformLayer);


}  // namespace caffe
