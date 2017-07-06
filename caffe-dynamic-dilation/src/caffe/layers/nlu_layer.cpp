#include <cmath>
#include <vector>

#include "caffe/layers/nlu_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
 NLUParameter nlu_param = this->layer_param_.nlu_param();
 t_ = 0;
 max_mean_ = nlu_param.max_mean();
 gamma_ = nlu_param.gamma();
 eta_ = nlu_param.eta();
 alpha_ = nlu_param.alpha();
 ratio_ = nlu_param.ratio();
}

template <typename Dtype>
void NLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  rand_vec_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void NLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();

  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min( std::max(bottom_data[i] * ratio_ + Dtype(0.5), Dtype(0)), Dtype(1));
  }
}

template <typename Dtype>
void NLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();

    Dtype* noise_gaussian =
          static_cast<Dtype*>(rand_vec_.mutable_gpu_data());
    Dtype fSTD = sqrt(eta_/pow(1+t_,gamma_));
    caffe_gpu_rng_gaussian(count, Dtype(max_mean_), fSTD, noise_gaussian);

    Dtype dThre = 1.0 / (2 * ratio_);

    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ( (std::abs(bottom_data[i]) > dThre) ? 0 : ratio_)
                     + ( (std::abs(bottom_data[i]) > dThre) ? (this->alpha_*(std::abs(bottom_data[i]) - dThre)*bottom_data[i]/std::abs(bottom_data[i])) : 0)
                     + noise_gaussian[i];
    }

    t_++;
  }
}

#ifdef CPU_ONLY
STUB_GPU(NLULayer);
#endif

INSTANTIATE_CLASS(NLULayer);
REGISTER_LAYER_CLASS(NLU);

}  // namespace caffe
