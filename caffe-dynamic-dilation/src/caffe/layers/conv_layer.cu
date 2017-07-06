#include <vector>
#include <iostream>

#include "caffe/layers/conv_layer.hpp"
#include "caffe/common.cuh"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/neuron_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bottom_dilation = NULL;

  if (bottom.size() == 2) {
    bottom_dilation = bottom[1]->gpu_data();
  }
  // Here, we modify original caffe to receive only one data input, the second optinal input is dilation matrix.
  for (int i = 0; i < 1; ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
     if (bottom.size() == 2) {
     for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_dynamic_dilation_gemm(bottom_data + n * this->bottom_dim_, bottom_dilation + bottom[1]->offset(n), weight,
            top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    } else {
      for (int n = 0; n < this->num_; ++n) {
        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
        }
      }
    }
  }
}

template <typename Dtype>
__global__ void DilationBackward(const unsigned int nthreads,
    const Dtype* data_cc, const Dtype* diff_top,
    const int output_num, const int height, const int width,
    Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int w = (index) % width;
    int h = (index / width) % height;

    atomicAdd(&bottom_diff[h * width + w], data_cc[index] * diff_top[index]);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_dilation = NULL;
  Dtype* bottom_dilation_diff = NULL;

  int height_ = this->output_shape_[0];
  int width_ = this->output_shape_[1];

  if (bottom.size() == 2) {
    bottom_dilation = bottom[1]->gpu_data();

    bottom_dilation_diff = bottom[1]->mutable_gpu_diff();
    caffe_gpu_set(this->num_ * height_ * width_, Dtype(0), bottom_dilation_diff);
  }

  for (int i = 0; i < 1; ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i] || this->force_back_propagation_) {
      if (bottom.size() == 2) {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_dynamic_dilation_gemm(bottom_data + n * this->bottom_dim_, bottom_dilation + bottom[1]->offset(n),
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i] || this->force_back_propagation_) {
            this->backward_gpu_dynamic_dilation_gemm(top_diff + n * this->top_dim_, bottom_dilation + bottom[1]->offset(n), weight, 
                bottom_diff + n * this->bottom_dim_);

            int kernel_h_ = this->kernel_shape_data[0];
            int kernel_w_ = this->kernel_shape_data[1];
            int stride_h_ = this->stride_data[0];
            int stride_w_ = this->stride_data[1];
            int pad_h_ = this->pad_data[0];
            int pad_w_ = this->pad_data[1];

            const int kernel_h_eff = kernel_h_ + (kernel_h_ - 1) * (pad_h_ - 1);
            const int kernel_w_eff = kernel_w_ + (kernel_w_ - 1) * (pad_w_ - 1);
            int height_col = (height_ + 2 * pad_h_ - kernel_h_eff) / stride_h_ + 1;
            int width_col = (width_ + 2 * pad_w_ - kernel_w_eff) / stride_w_ + 1;

            this->backward_gpu_dynamic_dilation_col2im_gemm(top_diff + n * this->top_dim_, bottom_data + bottom[0]->offset(n), bottom_dilation + bottom[1]->offset(n), weight,
              bottom_dilation_diff + bottom[1]->offset(n));
  
          }
        }
      } else {
        for (int n = 0; n < this->num_; ++n) {
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          if (this->param_propagate_down_[0]) {
            this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
                top_diff + n * this->top_dim_, weight_diff);
          }
          // gradient w.r.t. bottom data, if necessary.
          if (propagate_down[i] || this->force_back_propagation_) {
            this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
                bottom_diff + n * this->bottom_dim_);
          }
        }
      }
    }
  }
}
INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
