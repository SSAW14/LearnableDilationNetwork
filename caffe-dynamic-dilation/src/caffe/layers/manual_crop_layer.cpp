#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

//#include "caffe/common.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/syncedmem.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"

#include "caffe/layers/manual_crop_layer.hpp"

using namespace std;

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void ManualCropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ManualCropParameter manual_crop_param = this->layer_param_.manual_crop_param();
  x_ = manual_crop_param.x();
  y_ = manual_crop_param.y();
  h_ = manual_crop_param.h();
  w_ = manual_crop_param.w();
}

template <typename Dtype>
void ManualCropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = h_;
  width_ = w_;
  top[0]->Reshape(bottom[0]->num(), channels_, height_, width_);
}

template <typename Dtype>
void ManualCropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < top[0]->channels(); ++c) {
      for (int h = 0; h < top[0]->height(); ++h) {
        caffe_copy(top[0]->width(),
            bottom_data + bottom[0]->offset(n, c, x_ + h, y_),
            top_data + top[0]->offset(n, c, h));
      }
    }
  }
}

template <typename Dtype>
void ManualCropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < top[0]->channels(); ++c) {
        for (int h = 0; h < top[0]->height(); ++h) {
          caffe_copy(top[0]->width(),
              top_diff + top[0]->offset(n, c, h),
              bottom_diff + bottom[0]->offset(n, c, x_ + h, y_));
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ManualCropLayer);
#endif

INSTANTIATE_CLASS(ManualCropLayer);
REGISTER_LAYER_CLASS(ManualCrop);

}  // namespace caffe
