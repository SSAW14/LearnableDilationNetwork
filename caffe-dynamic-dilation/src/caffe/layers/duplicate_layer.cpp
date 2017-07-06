#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/duplicate_layer.hpp"

namespace caffe {

template <typename Dtype>
void DuplicateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DuplicateParameter duplicate_param = this->layer_param().duplicate_param();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = duplicate_param.height();
  width_ = duplicate_param.width();
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
/*
  for (int i = 0; i < count; ++i) {
    top_data[i] = bottom_data[i];
  }*/
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
/*
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i];
    }*/
  }
}


#ifdef CPU_ONLY
STUB_GPU(DuplicateLayer);
#endif

INSTANTIATE_CLASS(DuplicateLayer);
REGISTER_LAYER_CLASS(Duplicate);

}  // namespace caffe
