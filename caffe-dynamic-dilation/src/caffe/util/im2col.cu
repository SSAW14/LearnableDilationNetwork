#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/common.cuh"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// function of max(0,1-|x-x_grid|)
template <typename Dtype>
__device__ Dtype max_abs(const Dtype x, const Dtype x_grid) {
  if ((x_grid - x) * (x_grid - x) >= 1) {
    return 0;
  } else if (x_grid >= x) {
    return 1 - x_grid + x;
  } else {
    return 1 - x + x_grid;
  }
}

// derived function of max(0,1-|x-x_grid|)
template <typename Dtype>
__device__ Dtype diff_max_abs(const Dtype x, const Dtype x_grid) {
  if ((x_grid - x) * (x_grid - x) >= 1) {
    return 0;
  } else if (x_grid >= x) {
    return -1;
  } else {
    return 1;
  }
}

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    Dtype* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h -
      (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col = (width + 2 * pad_w -
      (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w, double* data_col);

template <typename Dtype>
__global__ void im2col_dynamic_gpu_kernel(const int n, const Dtype* data_im, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype* data_im_ceil_ceil = data_im;
    const Dtype* data_im_ceil_floor = data_im;
    const Dtype* data_im_floor_ceil = data_im;
    const Dtype* data_im_floor_floor = data_im;

    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;

    Dtype dilation = dilation_im[((n * channels + c_im) * height + h) * width + w];
    int iDilationFloor = static_cast<int>(dilation);
    int iDilationCeil = iDilationFloor + 1;
    if (iDilationFloor == pad_h) {
      iDilationFloor = iDilationFloor - 1;
      iDilationCeil = iDilationCeil - 1;
    }
    double dist = iDilationCeil - dilation;

    int h_im_ceil = h * stride_h - iDilationCeil;
    int w_im_ceil = w * stride_w - iDilationCeil;
    int h_im_floor = h * stride_h - iDilationFloor;
    int w_im_floor = w * stride_w - iDilationFloor;

    data_im_ceil_ceil += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_ceil; // patch start
    data_im_ceil_floor += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_floor; // patch start
    data_im_floor_ceil += ((n * channels + c_im) * height + h_im_floor) * width + w_im_ceil; // patch start
    data_im_floor_floor += ((n * channels + c_im) * height + h_im_floor) * width + w_im_floor; // patch start

    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im_ceil + i * iDilationCeil >= 0 && h_im_ceil + i * iDilationCeil < height && w_im_ceil + j * iDilationCeil >= 0 && w_im_ceil + j * iDilationCeil < width) {
	  *data_col = data_im_ceil_ceil[(i * iDilationCeil) * width + j * iDilationCeil] * (1-dist) * (1-dist) +
                      data_im_ceil_floor[(i * iDilationCeil) * width + j * iDilationFloor] * (1-dist) * dist +
                      data_im_floor_ceil[(i * iDilationFloor) * width + j * iDilationCeil] * dist * (1-dist) +
                      data_im_floor_floor[(i * iDilationFloor) * width + j * iDilationFloor] * dist * dist;
	}
	else {
	  *data_col = 0.;
	}
	data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_dynamic_gpu(const Dtype* data_im, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, Dtype* data_col) {
  // We are going to launch num * channels * height_col * width_col kernels, each
  // kernel responsible for copying a single block from a single image.
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (pad_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (pad_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_dynamic_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, dilation_im,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, 
      height_col, width_col,
      data_col);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_dynamic_gpu<float>(const float* data_im, const float* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, float* data_col);
template void im2col_dynamic_gpu<double>(const double* data_im, const double* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, double* data_col);

template <typename Dtype>
__global__ void im2col_dynamic_back_gpu_kernel(const int n, const Dtype* data_im, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int max_dilation, 
    const int stride_h, const int stride_w, const int height_col, const int width_col,
    Dtype* data_col, Dtype* data_dist) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype* data_im_ceil_ceil = data_im;
    const Dtype* data_im_ceil_floor = data_im;
    const Dtype* data_im_floor_ceil = data_im;
    const Dtype* data_im_floor_floor = data_im;

    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;

    Dtype dilation = dilation_im[h * width + w];
    int iDilationFloor = static_cast<int>(dilation);
    int iDilationCeil = iDilationFloor + 1;
    if (iDilationFloor == max_dilation) {
      iDilationFloor = iDilationFloor - 1;
      iDilationCeil = iDilationCeil - 1;
    }
//    Dtype dist = dilation - iDilationFloor;
    Dtype dist = iDilationCeil - dilation;
    if (c_im == 0) {
      data_dist[h * width + w] = dist;
    }

    int h_im_ceil = h * stride_h - iDilationCeil;
    int w_im_ceil = w * stride_w - iDilationCeil;
    int h_im_floor = h * stride_h - iDilationFloor;
    int w_im_floor = w * stride_w - iDilationFloor;

    data_im_ceil_ceil += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_ceil; // patch start
    data_im_ceil_floor += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_floor; // patch start
    data_im_floor_ceil += ((n * channels + c_im) * height + h_im_floor) * width + w_im_ceil; // patch start
    data_im_floor_floor += ((n * channels + c_im) * height + h_im_floor) * width + w_im_floor; // patch start

    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
//        Dtype diff_h = i-(kernel_h-1)/2;
//        Dtype diff_w = j-(kernel_w-1)/2;
	if (h_im_ceil + i * iDilationCeil >= 0 && h_im_ceil + i * iDilationCeil < height && w_im_ceil + j * iDilationCeil >= 0 && w_im_ceil + j * iDilationCeil < width) {
/*	  *data_col = data_im_ceil_ceil[(i * iDilationCeil) * width + j * iDilationCeil] * (diff_cc_h * diff_h  + diff_cc_w * diff_w) +
                      data_im_ceil_floor[(i * iDilationCeil) * width + j * iDilationFloor] * (diff_cf_h * diff_h  + diff_cf_w * diff_w) +
                      data_im_floor_ceil[(i * iDilationFloor) * width + j * iDilationCeil] * (diff_fc_h * diff_h  + diff_fc_w * diff_w) +
                      data_im_floor_floor[(i * iDilationFloor) * width + j * iDilationFloor] * (diff_ff_h * diff_h  + diff_ff_w * diff_w);*/

	  *data_col = data_im_ceil_ceil[(i * iDilationCeil) * width + j * iDilationCeil] * (2 - 2*dist) +
                      data_im_ceil_floor[(i * iDilationCeil) * width + j * iDilationFloor] * (2*dist - 1) +
                      data_im_floor_ceil[(i * iDilationFloor) * width + j * iDilationCeil] * (2*dist - 1) +
                      data_im_floor_floor[(i * iDilationFloor) * width + j * iDilationFloor] * (-2*dist);


	  //*data_col = 0;
	}
	else {
	  *data_col = 0;
	}
	data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void im2col_dynamic_back_gpu(const Dtype* data_im, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, Dtype* data_col) {
  // We are going to launch num * channels * height_col * width_col kernels, each
  // kernel responsible for copying a single block from a single image.
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (pad_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (pad_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)

  Blob<Dtype> dilation_temp;
  dilation_temp.Reshape(1, 1, height_col, width_col);
  Dtype* pDilation = dilation_temp.mutable_gpu_data();

  im2col_dynamic_back_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, dilation_im,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h,
      stride_h, stride_w, 
      height_col, width_col,
      data_col, pDilation);

  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_dynamic_back_gpu<float>(const float* data_im, const float* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, float* data_col);
template void im2col_dynamic_back_gpu<double>(const double* data_im, const double* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, double* data_col);

template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  __syncthreads();

  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % shared_col_shape[i + 1];
      channel_in /= shared_col_shape[i + 1];
      channel_out *= shared_kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= shared_col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * shared_stride[i] - shared_pad[i];
      channel_in *= shared_im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= shared_col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] * shared_dilation[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < shared_im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0] * shared_dilation[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= shared_im_shape[i + 1];
          data_im_offset += d_iter[i] * shared_dilation[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = shared_kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int num_kernels, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_col) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 2:
    im2col_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 4:
    im2col_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 5:
    im2col_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 6:
    im2col_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 7:
    im2col_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 8:
    im2col_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 9:
    im2col_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  case 10:
    im2col_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, dilation, data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    Dtype* data_im) {
  int height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) /
      stride_h + 1;
  int width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) /
      stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, const int dilation_h, const int dilation_w,
    double* data_im);

template <typename Dtype>
__global__ void elementwise_multiply_kernel(int count, const Dtype* input, Dtype* output) {
  CUDA_KERNEL_LOOP(index, count) {
    output[index] = input[index]*output[index];
  }
}

template <typename Dtype>
void elementwise_multiply(int count, const Dtype* input, Dtype* output) {
  elementwise_multiply_kernel<Dtype><<<CAFFE_GET_BLOCKS(count),
                             CAFFE_CUDA_NUM_THREADS>>>(count, input, output);

  CUDA_POST_KERNEL_CHECK;
}

template void elementwise_multiply(int count, const double* input, double* output);
template void elementwise_multiply(int count, const float* input, float* output);

template <typename Dtype>
__global__ void col2im_dynamic_gpu_kernel(const int n, const Dtype* data_col, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype* data_im_ceil_ceil = data_im;
    Dtype* data_im_ceil_floor = data_im;
    Dtype* data_im_floor_ceil = data_im;
    Dtype* data_im_floor_floor = data_im;

    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;

    Dtype dilation = dilation_im[((n * channels + c_im) * height + h) * width + w];
    int iDilationFloor = static_cast<int>(dilation);
    int iDilationCeil = iDilationFloor + 1;
    if (iDilationFloor == pad_h) {
      iDilationFloor = iDilationFloor - 1;
      iDilationCeil = iDilationCeil - 1;
    }
    Dtype dist = iDilationCeil - dilation;

    int h_im_ceil = h * stride_h - iDilationCeil;
    int w_im_ceil = w * stride_w - iDilationCeil;
    int h_im_floor = h * stride_h - iDilationFloor;
    int w_im_floor = w * stride_w - iDilationFloor;

    data_im_ceil_ceil += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_ceil; // patch start
    data_im_ceil_floor += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_floor; // patch start
    data_im_floor_ceil += ((n * channels + c_im) * height + h_im_floor) * width + w_im_ceil; // patch start
    data_im_floor_floor += ((n * channels + c_im) * height + h_im_floor) * width + w_im_floor; // patch start

    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im_ceil + i * iDilationCeil >= 0 && h_im_ceil + i * iDilationCeil < height && w_im_ceil + j * iDilationCeil >= 0 && w_im_ceil + j * iDilationCeil < width) {
	  atomicAdd(&data_im_ceil_ceil[(i * iDilationCeil) * width + j * iDilationCeil], (*data_col) * (1-dist) * (1-dist));
	  atomicAdd(&data_im_ceil_floor[(i * iDilationCeil) * width + j * iDilationFloor], (*data_col) * (1-dist) * dist);
	  atomicAdd(&data_im_floor_ceil[(i * iDilationFloor) * width + j * iDilationCeil], (*data_col) * dist * (1-dist));
	  atomicAdd(&data_im_floor_floor[(i * iDilationFloor) * width + j * iDilationFloor], (*data_col) * dist * dist);
	}
	data_col += height_col * width_col;
      }
    }
/*
    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;

    Dtype dilation = dilation_im[(n * height + h) * width + w];
    int iDilationFloor = static_cast<int>(dilation);
    if (dilation - iDilationFloor > 0.5) {
      dilation = iDilationFloor + 1;
    } else {
      dilation = iDilationFloor;
    }
    int dilation_h = dilation;
    int dilation_w = dilation;

    int h_im = h * stride_h - dilation_h;
    int w_im = w * stride_w - dilation_w;

    data_im += ((n * channels + c_im) * height + h_im) * width + w_im;
    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im + i * dilation_h >= 0 && h_im + i * dilation_h < height && w_im + j * dilation_w >= 0 && w_im + j * dilation_w < width) {
	  atomicAdd(&data_im[(i * dilation_h) * width + j * dilation_w], *data_col);
	}
	data_col += height_col * width_col;
      }
    }*/
  }
}

template <typename Dtype>
void col2im_dynamic_gpu(const Dtype* data_col, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, Dtype* data_im) {
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (pad_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (pad_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;

  caffe_gpu_set(num * channels * height * width, Dtype(0), data_im);
  col2im_dynamic_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, dilation_im,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, 
      height_col, width_col,
      data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_dynamic_gpu<float>(const float* data_col, const float* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, float* data_im);
template void col2im_dynamic_gpu<double>(const double* data_col, const double* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, double* data_im);

template <typename Dtype>
__global__ void col2im_dynamic_back_gpu_kernel(const int n, const Dtype* data_col, const Dtype* input_data, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;

    const Dtype* input_data_im_ceil_ceil = input_data; // input feature
    const Dtype* input_data_im_ceil_floor = input_data;
    const Dtype* input_data_im_floor_ceil = input_data;
    const Dtype* input_data_im_floor_floor = input_data;

    int w = index % width_col;
    index /= width_col;
    int h = index % height_col;
    index /= height_col;
    int c_im = index % channels;
    int n = index / channels;

    Dtype dilation = dilation_im[((n * channels + c_im) * height + h) * width + w];
    int iDilationFloor = static_cast<int>(dilation);
    int iDilationCeil = iDilationFloor + 1;
    if (iDilationFloor == pad_h) {
      iDilationFloor = iDilationFloor - 1;
      iDilationCeil = iDilationCeil - 1;
    }
    Dtype dist = iDilationCeil - dilation;

    int h_im_ceil = h * stride_h - iDilationCeil;
    int w_im_ceil = w * stride_w - iDilationCeil;
    int h_im_floor = h * stride_h - iDilationFloor;
    int w_im_floor = w * stride_w - iDilationFloor;

    input_data_im_ceil_ceil += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_ceil; // patch start
    input_data_im_ceil_floor += ((n * channels + c_im) * height + h_im_ceil) * width + w_im_floor; // patch start
    input_data_im_floor_ceil += ((n * channels + c_im) * height + h_im_floor) * width + w_im_ceil; // patch start
    input_data_im_floor_floor += ((n * channels + c_im) * height + h_im_floor) * width + w_im_floor; // patch start

    int channels_col = channels * kernel_h * kernel_w;
    int c = c_im * kernel_h * kernel_w;
    data_col += ((n * channels_col + c) * height_col + h) * width_col + w;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
	if (h_im_ceil + i * iDilationCeil >= 0 && h_im_ceil + i * iDilationCeil < height && w_im_ceil + j * iDilationCeil >= 0 && w_im_ceil + j * iDilationCeil < width) {
          val += (*data_col) * ((input_data_im_ceil_ceil[(i * iDilationCeil) * width + j * iDilationCeil]) * (2-2*dist) +
                (input_data_im_ceil_floor[(i * iDilationCeil) * width + j * iDilationFloor]) * (2*dist-1) +
                (input_data_im_floor_ceil[(i * iDilationFloor) * width + j * iDilationCeil]) * (2*dist-1) +
                (input_data_im_floor_floor[(i * iDilationFloor) * width + j * iDilationFloor]) * (-2*dist));
	}
	data_col += height_col * width_col;
      }
    }
    data_im[((n * channels + c_im) * height + h) * width + w] = val;
  }
}


template <typename Dtype>
void col2im_dynamic_back_gpu(const Dtype* data_col, const Dtype* input_data, const Dtype* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, Dtype* data_im) {
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (pad_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (pad_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int num_kernels = num * channels * height_col * width_col;

  caffe_gpu_set(num * channels * height * width, Dtype(0), data_im);
  col2im_dynamic_back_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, input_data, dilation_im,
      num, channels, height, width,
      kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, 
      height_col, width_col,
      data_im);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_dynamic_back_gpu<float>(const float* data_col, const float* input_data, const float* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, float* data_im);
template void col2im_dynamic_back_gpu<double>(const double* data_col, const double* input_data, const double* dilation_im,
    const int num, const int channels, const int height, const int width,
    const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
    const int stride_h, const int stride_w, double* data_im);

template <typename Dtype, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)

  __shared__ int shared_dilation[num_axes];
  __shared__ int shared_kernel_shape[num_axes];
  __shared__ int shared_pad[num_axes];
  __shared__ int shared_stride[num_axes];
  __shared__ int shared_col_shape[num_axes + 1];
  __shared__ int shared_im_shape[num_axes + 1];

  if (threadIdx.x < num_axes) {
    shared_dilation[threadIdx.x] = dilation[threadIdx.x];
    shared_kernel_shape[threadIdx.x] = kernel_shape[threadIdx.x];
    shared_pad[threadIdx.x] = pad[threadIdx.x];
    shared_stride[threadIdx.x] = stride[threadIdx.x];
  }
  if (threadIdx.x < num_axes + 1) {
    shared_col_shape[threadIdx.x] = col_shape[threadIdx.x];
    shared_im_shape[threadIdx.x] = im_shape[threadIdx.x];
  }
  __syncthreads();

  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % shared_im_shape[i + 1] + shared_pad[i];
      c_im /= shared_im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      const int kernel_extent =
          shared_dilation[i] * (shared_kernel_shape[i] - 1) + 1;
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_extent) ? 0 :
          (d_im[i] - kernel_extent) / shared_stride[i] + 1;
      d_col_end[i] =
          min(d_im[i] / shared_stride[i] + 1, shared_col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    bool skip = false;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      int kernel_index;
      for (int i = num_axes - 1; i >= 0; --i) {
        kernel_index = d_im[i] - d_col_iter[i] * shared_stride[i];
        if (kernel_index % shared_dilation[i]) {
          skip = true;
          break;
        } else {
          kernel_index /= shared_dilation[i];
          final_offset += kernel_index * kernel_shape_prod;
          kernel_shape_prod *= shared_kernel_shape[i];
        }
      }
      if (!skip) {
        final_offset += kernel_shape_prod * c_im;
        for (int i = 0; i < num_axes; ++i) {
          final_offset *= shared_col_shape[i + 1];
          final_offset += d_col_iter[i];
        }
        val += data_col[final_offset];
      }
      skip = false;
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, Dtype* data_im) {
  // num_axes should be smaller than block size
  DCHECK_LT(num_spatial_axes, CAFFE_CUDA_NUM_THREADS);
  switch (num_spatial_axes) {
  case 1:
    col2im_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 2:
    col2im_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 3:
    col2im_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 4:
    col2im_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 5:
    col2im_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 6:
    col2im_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 7:
    col2im_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 8:
    col2im_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 9:
    col2im_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  case 10:
    col2im_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, dilation, data_im);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    const int* dilation, double* data_im);

}  // namespace caffe
