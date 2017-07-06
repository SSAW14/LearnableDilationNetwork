#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/stop_layer.hpp"

namespace caffe {


template <typename Dtype>
void StopLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
}

template <typename Dtype>
void StopLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i];
  }
}

template <typename Dtype>
void StopLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(StopLayer);
#endif

INSTANTIATE_CLASS(StopLayer);
REGISTER_LAYER_CLASS(Stop);

}  // namespace caffe
