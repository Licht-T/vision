/*!
 ******************* BEGIN Caffe Copyright Notice and Disclaimer
 *****************
 *
 * COPYRIGHT
 *
 * All contributions by the University of California:
 * Copyright (c) 2014-2017 The Regents of the University of California (Regents)
 * All rights reserved.
 *
 * All other contributions:
 * Copyright (c) 2014-2017, the respective contributors
 * All rights reserved.
 *
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 *
 * LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * CONTRIBUTION AGREEMENT
 *
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 ***************** END Caffe Copyright Notice and Disclaimer
 *********************
 *
 * Copyright (c) 2018 Microsoft
 * Licensed under The MIT License [see LICENSE for details]
 * \file modulated_deformable_im2col.cuh
 * \brief Function definitions of converting an image to
 * column matrix based on kernel, padding, dilation, and offset.
 * These functions are mainly used in deformable convolution operators.
 * \ref: https://arxiv.org/abs/1703.06211
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai, Xizhou Zhu, Han Hu, Dazhi Cheng
 */

// modified from
// https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/blob/mmdetection/mmdet/ops/dcn/src/deform_conv_cuda_kernel.cu

// modified from
// https://github.com/open-mmlab/mmdetection/blob/master/mmdet/ops/dcn/src/deform_conv_cuda.cpp

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <TH/TH.h>

#include <cmath>
#include <iostream>
#include <tuple>

using namespace at;

const int kMaxParallelImgs = 32;

template <typename scalar_t>
static scalar_t bilinear_interpolate(
    const scalar_t* in,
    const int height,
    const int width,
    scalar_t h,
    scalar_t w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t>
static void deformable_im2col_kernel(
    const int n,
    const scalar_t* input,
    const scalar_t* offset,
    const scalar_t* mask,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dil_h,
    const int dil_w,
    const int batch_sz,
    const int n_in_channels,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    const bool use_mask,
    scalar_t* columns) {
  for (int index = 0; index != n; ++index) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    auto columns_ptr = columns +
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    auto input_ptr = input +
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    auto offset_ptr = offset +
        (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w * out_h *
            out_w;

    auto mask_ptr = mask;
    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
          out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const scalar_t offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w = offset_ptr
            [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t y = (out_y * stride_h - pad_h) + i * dil_h + offset_h;
        const scalar_t x = (out_x * stride_w - pad_w) + j * dil_w + offset_w;
        *columns_ptr =
            mask_value * bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

static void deformable_im2col(
    const at::Tensor input,
    const at::Tensor data_offset,
    const at::Tensor data_mask,
    int n_in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dil_h,
    int dil_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    at::Tensor data_col) {
  bool use_mask = data_mask.numel() != 0;
  int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "deformable_im2col", ([&] {
        deformable_im2col_kernel(
            num_kernels,
            input.data_ptr<scalar_t>(),
            data_offset.data_ptr<scalar_t>(),
            data_mask.data_ptr<scalar_t>(),
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            parallel_imgs,
            n_in_channels,
            deformable_group,
            out_h,
            out_w,
            use_mask,
            data_col.data_ptr<scalar_t>());
      }));
}

static int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

at::Tensor DeformConv2d_forward_cpu(
    const at::Tensor& input_param,
    const at::Tensor& weight_param,
    const at::Tensor& offset_param,
    const at::Tensor& mask_param,
    const at::Tensor& bias,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps) {
  at::Tensor input = input_param;
  at::Tensor offset = offset_param;
  at::Tensor mask = mask_param;
  at::Tensor weight = weight_param;

  bool use_mask = mask.numel() != 0;

  TORCH_CHECK(input.ndimension() == 4);
  TORCH_CHECK(offset.ndimension() == 4);
  TORCH_CHECK(!use_mask || mask.ndimension() == 4);
  TORCH_CHECK(weight.ndimension() == 4);
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(offset.is_contiguous());
  TORCH_CHECK(mask.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(input.device().is_cpu(), "input must be a CPU tensor");

  int batch_sz = input.size(0);
  int n_in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  // Unpack shapes and args
  int out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  int ker_h = dil_h * (weight_h - 1) + 1;
  int ker_w = dil_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(
      weight_h > 0 && weight_w > 0,
      "weight_h: ",
      weight_h,
      " weight_w: ",
      weight_w);
  TORCH_CHECK(
      stride_h > 0 && stride_w > 0,
      "stride_h: ",
      stride_h,
      " stride_w: ",
      stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(dil_h > 0 && dil_w > 0, "dil_h: ", dil_h, " dil_w: ", dil_w);

  TORCH_CHECK(weight.size(1) * n_weight_grps == input.size(1));
  TORCH_CHECK(weight.size(0) % n_weight_grps == 0);
  TORCH_CHECK(
      (offset.size(1) == n_offset_grps * 2 * weight_h * weight_w),
      "offset.shape[1] is not valid: got: ",
      offset.size(1),
      " expected: ",
      n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ",
      mask.size(1),
      " expected: ",
      n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(input.size(1) % n_offset_grps == 0);

  TORCH_CHECK(
      (offset.size(0) == input.size(0)), "invalid batch size of offset");
  TORCH_CHECK(
      (offset.size(2) == out_h && offset.size(3) == out_w),
      "offset output dims: (",
      offset.size(2),
      ", ",
      offset.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK((mask.size(0) == input.size(0)), "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask.size(2) == out_h && mask.size(3) == out_w)),
      "offset output dims: (",
      mask.size(2),
      ", ",
      mask.size(3),
      ") - ",
      "computed output dims: (",
      out_h,
      ", ",
      out_w,
      ")");
  TORCH_CHECK(
      out_h > 0 && out_w > 0,
      "Calculated output size too small - out_h: ",
      out_h,
      " out_w: ",
      out_w);

  auto out = at::zeros({batch_sz, out_channels, out_h, out_w}, input.options());

  // Separate batches into blocks
  out = out.view({batch_sz / n_parallel_imgs,
                  n_parallel_imgs,
                  out_channels,
                  out_h,
                  out_w});
  input = input.view(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  offset = offset.view({batch_sz / n_parallel_imgs,
                        n_parallel_imgs,
                        n_offset_grps * 2 * weight_h * weight_w,
                        out_h,
                        out_w});

  if (use_mask) {
    mask = mask.view({batch_sz / n_parallel_imgs,
                      n_parallel_imgs,
                      n_offset_grps * weight_h * weight_w,
                      out_h,
                      out_w});
  }

  at::Tensor out_buf = at::zeros(
      {batch_sz / n_parallel_imgs,
       out_channels,
       n_parallel_imgs * out_h,
       out_w},
      out.options());

  // Separate channels into convolution groups
  out_buf = out_buf.view({out_buf.size(0),
                          n_weight_grps,
                          out_buf.size(1) / n_weight_grps,
                          out_buf.size(2),
                          out_buf.size(3)});
  weight = weight.view({n_weight_grps,
                        weight.size(0) / n_weight_grps,
                        weight.size(1),
                        weight.size(2),
                        weight.size(3)});

  // Sample points and perform convolution
  auto columns = at::zeros(
      {n_in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w},
      input.options());
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col(
        input[b],
        offset[b],
        mask[b],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        columns);

    columns = columns.view(
        {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      out_buf[b][g] = out_buf[b][g]
                          .flatten(1)
                          .addmm_(weight[g].flatten(1), columns[g])
                          .view_as(out_buf[b][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  out_buf = out_buf.view({batch_sz / n_parallel_imgs,
                          out_channels,
                          n_parallel_imgs,
                          out_h,
                          out_w});
  out_buf.transpose_(1, 2);
  out.copy_(out_buf);
  out = out.view({batch_sz, out_channels, out_h, out_w});

  return out + bias.view({1, out_channels, 1, 1});
}

template <typename scalar_t>
static void deformable_col2im_kernel(
    const int n,
    const scalar_t* col,
    const scalar_t* offset,
    const scalar_t* mask,
    const int channels,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_sz,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    const bool use_mask,
    scalar_t* grad_im) {
  for (int index = 0; index != n; ++index) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int b = (index / (out_w * out_h)) % batch_sz;
    const int j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const int i = (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const int c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    int c_per_offset_grp = channels / n_offset_grps;
    const int offset_grp = c / c_per_offset_grp;

    auto offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w * out_h *
            out_w;

    auto mask_ptr = mask;
    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * kernel_h * kernel_w *
          out_h * out_w;
    }

    const int mask_idx = i * kernel_w + j;
    const int offset_idx = 2 * mask_idx;

    const int offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const int offset_w_ptr = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const scalar_t offset_h = offset_ptr[offset_h_ptr];
    const scalar_t offset_w = offset_ptr[offset_w_ptr];

    scalar_t mask_value = 1;
    if (use_mask) {
      mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
    }

    const scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int yp = int(y) + dy;
        int xp = int(x) + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width &&
            std::abs(y - yp) < 1 && std::abs(x - xp) < 1) {
          int grad_pos = ((b * channels + c) * height + yp) * width + xp;
          scalar_t weight = (1 - std::abs(y - yp)) * (1 - std::abs(x - xp));
          grad_im[grad_pos] += mask_value * weight * col[index];
        }
      }
    }
  }
}

static void compute_grad_input(
    const at::Tensor columns,
    const at::Tensor offset,
    const at::Tensor mask,
    const int channels,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int n_offset_grps,
    at::Tensor grad_im) {
  bool use_mask = mask.numel() != 0;

  int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * weight_h * weight_w * out_h * out_w * parallel_imgs;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "deformable_col2im", ([&] {
        deformable_col2im_kernel(
            num_kernels,
            columns.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_im.data_ptr<scalar_t>());
      }));
}

template <typename scalar_t>
static scalar_t get_coordinate_weight(
    const scalar_t* im_data,
    const int height,
    const int width,
    scalar_t y,
    scalar_t x,
    bool is_y_direction) {
  int y_l = floor(y);
  int x_l = floor(x);
  int y_h = y_l + 1;
  int x_h = x_l + 1;

  bool valid_y_l = 0 <= y_l && y_l < height;
  bool valid_y_h = 0 <= y_h && y_h < height;
  bool valid_x_l = 0 <= x_l && x_l < width;
  bool valid_x_h = 0 <= x_h && x_h < width;

  scalar_t zero = 0;
  scalar_t v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
  scalar_t v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
  scalar_t v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
  scalar_t v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

  if (is_y_direction) {
    scalar_t dx = x - x_l;
    return dx * (v_YX - v_yX) + (1 - dx) * (v_Yx - v_yx);
  } else {
    scalar_t dy = y - y_l;
    return dy * (v_YX - v_Yx) + (1 - dy) * (v_yX - v_yx);
  }
}

template <typename scalar_t>
static void deformable_col2im_coord_kernel(
    const int n,
    const scalar_t* col,
    const scalar_t* im,
    const scalar_t* offset,
    const scalar_t* mask,
    const int channels,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int batch_sz,
    const int offset_channels,
    const int n_offset_grps,
    const int out_h,
    const int out_w,
    const bool use_mask,
    scalar_t* grad_offset,
    scalar_t* grad_mask) {
  for (int index = 0; index != n; ++index) {
    scalar_t grad_offset_val = 0;
    scalar_t grad_mask_val = 0;

    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int w_w = (index / (out_w * out_h * 2)) % weight_w;
    int w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);

    const int offset_grp = c / (2 * weight_h * weight_w);
    const int col_step = weight_h * weight_w;

    int c_per_offset_grp = channels / n_offset_grps;

    auto col_ptr = col +
        offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz * out_w *
            out_h;
    auto im_ptr = im +
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    auto offset_ptr = offset +
        (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w * out_h *
            out_w;

    auto mask_ptr = mask;
    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * weight_h * weight_w *
          out_h * out_w;
    }

    const int offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const int is_y_direction = offset_c % 2 == 0;

    const int c_bound = c_per_offset_grp * weight_h * weight_w;
    for (int col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const int col_pos = (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      int out_x = col_pos % out_w;
      int out_y = (col_pos / out_w) % out_h;
      int j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      int i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const int mask_idx = i * weight_w + j;

      const int offset_h_idx =
          (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const int offset_w_idx =
          (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const scalar_t offset_h = offset_ptr[offset_h_idx];
      const scalar_t offset_w = offset_ptr[offset_w_idx];

      scalar_t mask_value = 1;
      if (use_mask) {
        mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
      }

      scalar_t y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      scalar_t x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const scalar_t weight =
          get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] *
            bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const int idx =
          ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w +
            w_w) *
               out_h +
           h) *
              out_w +
          w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

static void compute_grad_offset_and_mask(
    const at::Tensor columns,
    const at::Tensor input,
    const at::Tensor offset,
    const at::Tensor mask,
    const int channels,
    const int height,
    const int width,
    const int weight_h,
    const int weight_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int parallel_imgs,
    const int n_offset_grps,
    at::Tensor grad_offset,
    at::Tensor grad_mask) {
  bool use_mask = mask.numel() != 0;

  int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      out_h * out_w * 2 * weight_h * weight_w * n_offset_grps * parallel_imgs;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "deformable_col2im_coord", ([&] {
        deformable_col2im_coord_kernel(
            num_kernels,
            columns.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            offset.data_ptr<scalar_t>(),
            mask.data_ptr<scalar_t>(),
            channels,
            height,
            width,
            weight_h,
            weight_w,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            parallel_imgs,
            2 * weight_h * weight_w * n_offset_grps,
            n_offset_grps,
            out_h,
            out_w,
            use_mask,
            grad_offset.data_ptr<scalar_t>(),
            grad_mask.data_ptr<scalar_t>());
      }));
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor>
deform_conv2d_backward_input_cpu(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor grad_out,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs) {
  bool use_mask = mask.numel() != 0;

  int batch_sz = input.size(0);
  int n_in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  long out_h = (in_h + 2 * pad_h - (dil_h * (weight_h - 1) + 1)) / stride_h + 1;
  long out_w = (in_w + 2 * pad_w - (dil_w * (weight_w - 1) + 1)) / stride_w + 1;

  auto grad_input = at::zeros_like(input);
  auto grad_offset = at::zeros_like(offset);
  auto grad_mask = at::zeros_like(mask);
  auto columns = at::empty(
      {n_in_channels * weight_w * weight_h, n_parallel_imgs * out_h * out_w},
      input.options());

  // Separate into blocks
  grad_input = grad_input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  grad_offset = grad_offset.reshape({batch_sz / n_parallel_imgs,
                                     n_parallel_imgs,
                                     n_offset_grps * 2 * weight_h * weight_w,
                                     out_h,
                                     out_w});
  offset = offset.reshape({batch_sz / n_parallel_imgs,
                           n_parallel_imgs,
                           n_offset_grps * 2 * weight_h * weight_w,
                           out_h,
                           out_w});

  if (use_mask) {
    grad_mask = grad_mask.reshape({batch_sz / n_parallel_imgs,
                                   n_parallel_imgs,
                                   n_offset_grps * weight_h * weight_w,
                                   out_h,
                                   out_w});
    mask = mask.reshape({batch_sz / n_parallel_imgs,
                         n_parallel_imgs,
                         n_offset_grps * weight_h * weight_w,
                         out_h,
                         out_w});
  }

  grad_out = grad_out
                 .reshape({batch_sz / n_parallel_imgs,
                           n_parallel_imgs,
                           n_weight_grps,
                           n_out_channels / n_weight_grps,
                           out_h,
                           out_w})
                 .permute({0, 2, 3, 1, 4, 5});

  weight = weight.reshape({n_weight_grps,
                           weight.size(0) / n_weight_grps,
                           weight.size(1),
                           weight.size(2),
                           weight.size(3)});

  columns = columns.view(
      {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    columns.zero_();
    // Separate into weight groups
    for (int g = 0; g < n_weight_grps; g++) {
      columns[g] = columns[g].addmm_(
          weight[g].flatten(1).transpose(0, 1), grad_out[elt][g].flatten(1));
    }

    compute_grad_offset_and_mask(
        columns,
        input[elt],
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        n_parallel_imgs,
        n_offset_grps,
        grad_offset[elt],
        grad_mask[elt]);

    compute_grad_input(
        columns,
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        n_parallel_imgs,
        n_offset_grps,
        grad_input[elt]);
  }

  grad_input = grad_input.view({batch_sz, n_in_channels, in_h, in_w});
  grad_offset = grad_offset.view(
      {batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    grad_mask = grad_mask.view(
        {batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  return std::make_tuple(grad_input, grad_offset, grad_mask);
}

static at::Tensor deform_conv2d_backward_parameters_cpu(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor offset,
    at::Tensor mask,
    at::Tensor grad_out,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps,
    int n_parallel_imgs) {
  bool use_mask = mask.numel() != 0;

  int batch_sz = input.size(0);
  int n_in_channels = input.size(1);
  int in_h = input.size(2);
  int in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  int stride_h = stride.first;
  int stride_w = stride.second;

  int pad_h = pad.first;
  int pad_w = pad.second;

  int dil_h = dilation.first;
  int dil_w = dilation.second;

  long out_h = grad_out.size(2);
  long out_w = grad_out.size(3);

  auto grad_weight = at::zeros_like(weight);

  at::Tensor grad_out_buf = grad_out
                                .reshape({batch_sz / n_parallel_imgs,
                                          n_parallel_imgs,
                                          n_weight_grps,
                                          n_out_channels / n_weight_grps,
                                          out_h,
                                          out_w})
                                .permute({0, 2, 3, 1, 4, 5})
                                .contiguous();

  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  offset = offset.reshape({batch_sz / n_parallel_imgs,
                           n_parallel_imgs,
                           n_offset_grps * 2 * weight_h * weight_w,
                           out_h,
                           out_w});

  if (use_mask) {
    mask = mask.reshape({batch_sz / n_parallel_imgs,
                         n_parallel_imgs,
                         n_offset_grps * weight_h * weight_w,
                         out_h,
                         out_w});
  }

  grad_weight = grad_weight.view({n_weight_grps,
                                  grad_weight.size(0) / n_weight_grps,
                                  grad_weight.size(1),
                                  grad_weight.size(2),
                                  grad_weight.size(3)});

  auto columns = at::empty(
      {n_weight_grps,
       n_in_channels * weight_w * weight_h / n_weight_grps,
       n_parallel_imgs * out_h * out_w},
      input.options());

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    deformable_im2col(
        input[elt],
        offset[elt],
        mask[elt],
        n_in_channels,
        in_h,
        in_w,
        weight_h,
        weight_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dil_h,
        dil_w,
        out_h,
        out_w,
        n_parallel_imgs,
        n_offset_grps,
        columns);

    for (int g = 0; g < n_weight_grps; g++) {
      grad_weight[g] =
          grad_weight[g]
              .flatten(1)
              .addmm_(
                  grad_out_buf[elt][g].flatten(1), columns[g].transpose(1, 0))
              .view_as(grad_weight[g]);
    }
  }

  grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                  grad_weight.size(2),
                                  grad_weight.size(3),
                                  grad_weight.size(4)});
  return grad_weight;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
DeformConv2d_backward_cpu(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& offset,
    const at::Tensor& mask,
    const at::Tensor& bias,
    std::pair<int, int> stride,
    std::pair<int, int> pad,
    std::pair<int, int> dilation,
    int n_weight_grps,
    int n_offset_grps) {
  const int batch_sz = input.size(0);
  const int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  auto grad_input_and_offset_and_mask = deform_conv2d_backward_input_cpu(
      input,
      weight,
      offset,
      mask,
      grad_out,
      stride,
      pad,
      dilation,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs);

  auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
  auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
  auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

  auto grad_weight = deform_conv2d_backward_parameters_cpu(
      input,
      weight,
      offset,
      mask,
      grad_out,
      stride,
      pad,
      dilation,
      n_weight_grps,
      n_offset_grps,
      n_parallel_imgs);

  auto grad_bias = at::ones_like(bias) * grad_out.sum({0, 2, 3});

  return std::make_tuple(
      grad_input, grad_weight, grad_offset, grad_mask, grad_bias);
}
