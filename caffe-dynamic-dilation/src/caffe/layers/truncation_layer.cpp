#include <cmath>
#include <vector>

#include "caffe/layers/truncation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TruncationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 TruncationParameter truncation_param = this->layer_param_.truncation_param();

 truncation_min_ = truncation_param.truncation_min();
 truncation_max_ = truncation_param.truncation_max();
}

template <typename Dtype>
void TruncationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; ++i) {
/*
    if (bottom_data[i] < truncation_min_) {
      top_data[i] = truncation_min_;
    } else if (bottom_data[i] > truncation_max_) {
      top_data[i] = truncation_max_;
    } else {
      top_data[i] = bottom_data[i];
    }*/
    top_data[i] = std::min( std::max(bottom_data[i], Dtype(truncation_min_)), Dtype(truncation_max_));
  }
}

template <typename Dtype>
void TruncationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    for (int i = 0; i < count; ++i) {
      if (bottom_data[i] < truncation_min_) {
        bottom_diff[i] = std::max(top_diff[i], Dtype(0));
      } else if (bottom_data[i] > truncation_max_) {
        bottom_diff[i] = std::min(top_diff[i], Dtype(0));
      } else {
        bottom_diff[i] = top_diff[i];
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TruncationLayer);
#endif

INSTANTIATE_CLASS(TruncationLayer);
REGISTER_LAYER_CLASS(Truncation);

}  // namespace caffe
