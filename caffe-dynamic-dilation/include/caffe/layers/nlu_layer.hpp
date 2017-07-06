#ifndef CAFFE_NLU_LAYER_HPP_
#define CAFFE_NLU_LAYER_HPP_

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
class NLULayer : public NeuronLayer<Dtype> {
 public:
  explicit NLULayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NLU"; }

 protected:
  /**
   * @param 
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the nlu inputs.
   *
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int t_;
  float max_mean_;
  float gamma_;
  float eta_;
  float alpha_;
  float ratio_;

  Blob<Dtype> rand_vec_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_LAYER_HPP_
