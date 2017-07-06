#ifndef CAFFE_TRUNCATION_LAYER_HPP_
#define CAFFE_TRUNCATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Noise linear unit @f$

 * Note that the gradient vanishes as the values move away from 0.
 * The NLULayer is often a better choice for this reason.
 */
template <typename Dtype>
class TruncationLayer : public NeuronLayer<Dtype> {
 public:
  explicit TruncationLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Truncation"; }

 protected:
  /**
   * @param 
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float truncation_min_;
  float truncation_max_;

};

}  // namespace caffe

#endif  // CAFFE_TRUNCATION_LAYER_HPP_
