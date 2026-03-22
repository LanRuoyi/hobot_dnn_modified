// Copyright (c) 2024，Horizon Robotics.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "dnn_node/util/output_parser/detection/ptq_yolo11_seg_output_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dnn_node/util/output_parser/utils.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_yolo11_seg {

namespace {

struct PTQYolo11SegConfig {
  std::vector<int> strides;
  int reg_dim;
  int class_num;
  int mask_channels;
  std::vector<std::string> class_names;
};

PTQYolo11SegConfig yolo11_seg_config_ = {
    {8, 16, 32},
    16,
    1,
    32,
    {"road"}};

float score_threshold_ = 0.25f;
float nms_threshold_ = 0.7f;
int nms_top_k_ = 300;

template <typename T>
float ReadScalar(const T *data, size_t offset) {
  return static_cast<float>(data[offset]);
}

template <>
float ReadScalar<float>(const float *data, size_t offset) {
  return data[offset];
}

int InitClassNum(const int class_num) {
  if (class_num <= 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  yolo11_seg_config_.class_num = class_num;
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (!fi) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }

  std::vector<std::string> names;
  std::string line;
  while (std::getline(fi, line)) {
    if (!line.empty()) {
      names.emplace_back(line);
    }
  }

  if (names.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "cls name file %s is empty",
                 cls_name_file.c_str());
    return -1;
  }

  if (static_cast<int>(names.size()) != yolo11_seg_config_.class_num) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "class_names length %d is not equal to class_num %d",
                 static_cast<int>(names.size()),
                 yolo11_seg_config_.class_num);
    return -1;
  }

  yolo11_seg_config_.class_names.swap(names);
  return 0;
}

int InitStrides(const std::vector<int> &strides, int model_output_count) {
  if (static_cast<int>(strides.size()) != model_output_count) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "strides size %d is not equal to model_output_count %d",
                 static_cast<int>(strides.size()),
                 model_output_count);
    return -1;
  }
  yolo11_seg_config_.strides = strides;
  return 0;
}

int ParseTensorToFloat(const std::shared_ptr<DNNTensor> &tensor,
                       int batch_index,
                       std::vector<float> &data,
                       int &height,
                       int &width,
                       int &channel) {
  if (!tensor) {
    return -1;
  }

  hbSysFlushMem(&(tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

  int h_index = 0;
  int w_index = 0;
  int c_index = 0;
  if (hobot::dnn_node::output_parser::TensorUtils::GetTensorHWCIndex(
          tensor->properties.tensorLayout,
          &h_index,
          &w_index,
          &c_index) != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "unsupported tensor layout: %d",
                 tensor->properties.tensorLayout);
    return -1;
  }

  const int n = tensor->properties.validShape.dimensionSize[0];
  if (batch_index < 0 || batch_index >= n) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "invalid batch_index: %d, batch: %d",
                 batch_index,
                 n);
    return -1;
  }

  height = tensor->properties.validShape.dimensionSize[h_index];
  width = tensor->properties.validShape.dimensionSize[w_index];
  channel = tensor->properties.validShape.dimensionSize[c_index];

  const int aligned_h = tensor->properties.alignedShape.dimensionSize[h_index];
  const int aligned_w = tensor->properties.alignedShape.dimensionSize[w_index];
  const int aligned_c = tensor->properties.alignedShape.dimensionSize[c_index];

  if (height <= 0 || width <= 0 || channel <= 0 || aligned_h <= 0 ||
      aligned_w <= 0 || aligned_c <= 0) {
    return -1;
  }

  const size_t aligned_count = static_cast<size_t>(n) * aligned_h * aligned_w *
                               aligned_c;
  if (aligned_count == 0) {
    return -1;
  }

  const size_t mem_size = tensor->sysMem[0].memSize;
  size_t elem_size = mem_size / aligned_count;
  if (elem_size == 0) {
    elem_size = sizeof(float);
  }

  std::vector<float> scales;
  if (tensor->properties.quantiType != hbDNNQuantiType::NONE) {
    hobot::dnn_node::output_parser::TensorUtils::GetTensorScale(
        tensor->properties,
        scales);
    if (scales.empty()) {
      scales.push_back(1.0f);
    }
  }

  auto get_scale = [&scales](int c) {
    if (scales.empty()) {
      return 1.0f;
    }
    if (scales.size() == 1) {
      return scales[0];
    }
    if (c < 0 || static_cast<size_t>(c) >= scales.size()) {
      return scales.back();
    }
    return scales[static_cast<size_t>(c)];
  };

  data.resize(static_cast<size_t>(height) * width * channel);
  const bool is_nhwc = (tensor->properties.tensorLayout == HB_DNN_LAYOUT_NHWC);

  auto write_value = [&](int h, int w, int c, float value) {
    const size_t out_offset = (static_cast<size_t>(h) * width + w) * channel + c;
    data[out_offset] = value;
  };

  if (tensor->properties.quantiType == hbDNNQuantiType::NONE) {
    const auto *base = reinterpret_cast<const float *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          size_t in_offset = 0;
          if (is_nhwc) {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_h + h) *
                             aligned_w +
                         w) *
                            aligned_c +
                        c;
          } else {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_c + c) *
                             aligned_h +
                         h) *
                            aligned_w +
                        w;
          }
          write_value(h, w, c, ReadScalar(base, in_offset));
        }
      }
    }
    return 0;
  }

  if (elem_size == 1) {
    const auto *base = reinterpret_cast<const int8_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          size_t in_offset = 0;
          if (is_nhwc) {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_h + h) *
                             aligned_w +
                         w) *
                            aligned_c +
                        c;
          } else {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_c + c) *
                             aligned_h +
                         h) *
                            aligned_w +
                        w;
          }
          write_value(h, w, c, static_cast<float>(base[in_offset]) * get_scale(c));
        }
      }
    }
  } else if (elem_size == 2) {
    const auto *base = reinterpret_cast<const int16_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          size_t in_offset = 0;
          if (is_nhwc) {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_h + h) *
                             aligned_w +
                         w) *
                            aligned_c +
                        c;
          } else {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_c + c) *
                             aligned_h +
                         h) *
                            aligned_w +
                        w;
          }
          write_value(h, w, c, static_cast<float>(base[in_offset]) * get_scale(c));
        }
      }
    }
  } else {
    const auto *base = reinterpret_cast<const int32_t *>(tensor->sysMem[0].virAddr);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int c = 0; c < channel; ++c) {
          size_t in_offset = 0;
          if (is_nhwc) {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_h + h) *
                             aligned_w +
                         w) *
                            aligned_c +
                        c;
          } else {
            in_offset = ((static_cast<size_t>(batch_index) * aligned_c + c) *
                             aligned_h +
                         h) *
                            aligned_w +
                        w;
          }
          write_value(h, w, c, static_cast<float>(base[in_offset]) * get_scale(c));
        }
      }
    }
  }

  return 0;
}

struct Yolo11SegBranch {
  int stride = 0;
  int height = 0;
  int width = 0;
  std::vector<float> cls;
  std::vector<float> bbox;
  std::vector<float> coeff;
};

struct Yolo11SegCandidate {
  Detection det;
  std::vector<float> coeff;
};

std::vector<float> Softmax(const float *input, int length) {
  std::vector<float> output(static_cast<size_t>(length), 0.0f);
  float max_value = input[0];
  for (int i = 1; i < length; ++i) {
    if (input[i] > max_value) {
      max_value = input[i];
    }
  }

  float sum = 0.0f;
  for (int i = 0; i < length; ++i) {
    output[static_cast<size_t>(i)] = std::exp(input[i] - max_value);
    sum += output[static_cast<size_t>(i)];
  }

  if (sum <= std::numeric_limits<float>::epsilon()) {
    return output;
  }

  for (int i = 0; i < length; ++i) {
    output[static_cast<size_t>(i)] /= sum;
  }
  return output;
}

void NmsWithIndex(const std::vector<Yolo11SegCandidate> &input,
                  float iou_threshold,
                  int top_k,
                  std::vector<size_t> &selected_index) {
  selected_index.clear();
  if (input.empty()) {
    return;
  }

  std::vector<size_t> order(input.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }

  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return input[a].det.score > input[b].det.score;
  });

  std::vector<bool> suppressed(order.size(), false);
  int count = 0;

  for (size_t i = 0; i < order.size() && count < top_k; ++i) {
    if (suppressed[i]) {
      continue;
    }

    const auto &det_i = input[order[i]].det;
    selected_index.push_back(order[i]);
    ++count;

    const float area_i = std::max(0.0f, det_i.bbox.xmax - det_i.bbox.xmin) *
                         std::max(0.0f, det_i.bbox.ymax - det_i.bbox.ymin);

    for (size_t j = i + 1; j < order.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }

      const auto &det_j = input[order[j]].det;
      if (det_i.id != det_j.id) {
        continue;
      }

      const float xx1 = std::max(det_i.bbox.xmin, det_j.bbox.xmin);
      const float yy1 = std::max(det_i.bbox.ymin, det_j.bbox.ymin);
      const float xx2 = std::min(det_i.bbox.xmax, det_j.bbox.xmax);
      const float yy2 = std::min(det_i.bbox.ymax, det_j.bbox.ymax);

      if (xx2 <= xx1 || yy2 <= yy1) {
        continue;
      }

      const float area_j = std::max(0.0f, det_j.bbox.xmax - det_j.bbox.xmin) *
                           std::max(0.0f, det_j.bbox.ymax - det_j.bbox.ymin);
      const float intersection = (xx2 - xx1) * (yy2 - yy1);
      const float union_area = area_i + area_j - intersection;
      if (union_area <= std::numeric_limits<float>::epsilon()) {
        continue;
      }

      const float iou = intersection / union_area;
      if (iou > iou_threshold) {
        suppressed[j] = true;
      }
    }
  }
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception,
                int batch_index) {
  if (batch_index != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "YOLO11-seg only supports batch=1, but batch_index=%d",
                 batch_index);
    return -1;
  }

  if (output_tensors.size() < 10) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "invalid output tensor size: %zu, expect >= 10",
                 output_tensors.size());
    return -1;
  }

  std::vector<int> cls_indices;
  std::vector<int> bbox_indices;
  std::vector<int> coeff_indices;
  int proto_index = -1;
  int proto_hw = -1;

  const int class_num = yolo11_seg_config_.class_num;
  const int reg_dim = yolo11_seg_config_.reg_dim;
  const int mask_channels = yolo11_seg_config_.mask_channels;

  for (size_t i = 0; i < output_tensors.size(); ++i) {
    int h = 0;
    int w = 0;
    int c = 0;
    if (hobot::dnn_node::output_parser::TensorUtils::GetTensorValidHWC(
            &(output_tensors[i]->properties),
            &h,
            &w,
            &c) != 0) {
      continue;
    }

    if (c == class_num) {
      cls_indices.push_back(static_cast<int>(i));
      continue;
    }
    if (c == reg_dim * 4) {
      bbox_indices.push_back(static_cast<int>(i));
      continue;
    }
    if (c == mask_channels) {
      const int hw = h * w;
      if (hw > proto_hw) {
        if (proto_index >= 0) {
          coeff_indices.push_back(proto_index);
        }
        proto_hw = hw;
        proto_index = static_cast<int>(i);
      } else {
        coeff_indices.push_back(static_cast<int>(i));
      }
    }
  }

  if (cls_indices.size() != 3 || bbox_indices.size() != 3 ||
      coeff_indices.size() != 3 || proto_index < 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "can not map yolo11-seg outputs, cls=%zu bbox=%zu coeff=%zu proto=%d",
                 cls_indices.size(),
                 bbox_indices.size(),
                 coeff_indices.size(),
                 proto_index);
    return -1;
  }

  std::vector<Yolo11SegBranch> branches;
  branches.reserve(3);

  for (size_t i = 0; i < cls_indices.size(); ++i) {
    std::vector<float> cls_data;
    std::vector<float> bbox_data;
    std::vector<float> coeff_data;
    int cls_h = 0;
    int cls_w = 0;
    int cls_c = 0;
    int bbox_h = 0;
    int bbox_w = 0;
    int bbox_c = 0;
    int coeff_h = 0;
    int coeff_w = 0;
    int coeff_c = 0;

    if (ParseTensorToFloat(output_tensors[cls_indices[i]],
                           batch_index,
                           cls_data,
                           cls_h,
                           cls_w,
                           cls_c) != 0 ||
        ParseTensorToFloat(output_tensors[bbox_indices[i]],
                           batch_index,
                           bbox_data,
                           bbox_h,
                           bbox_w,
                           bbox_c) != 0 ||
        ParseTensorToFloat(output_tensors[coeff_indices[i]],
                           batch_index,
                           coeff_data,
                           coeff_h,
                           coeff_w,
                           coeff_c) != 0) {
      return -1;
    }

    if (cls_h != bbox_h || cls_w != bbox_w || cls_h != coeff_h ||
        cls_w != coeff_w || cls_c != class_num || bbox_c != reg_dim * 4 ||
        coeff_c != mask_channels) {
      RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                   "branch shape mismatch, cls(%d,%d,%d) bbox(%d,%d,%d) coeff(%d,%d,%d)",
                   cls_h,
                   cls_w,
                   cls_c,
                   bbox_h,
                   bbox_w,
                   bbox_c,
                   coeff_h,
                   coeff_w,
                   coeff_c);
      return -1;
    }

    Yolo11SegBranch branch;
    branch.stride = 0;
    branch.height = cls_h;
    branch.width = cls_w;
    branch.cls.swap(cls_data);
    branch.bbox.swap(bbox_data);
    branch.coeff.swap(coeff_data);
    branches.emplace_back(std::move(branch));
  }

  // 按特征图分辨率从高到低排序，便于和strides一一对应
  std::sort(branches.begin(), branches.end(), [](const Yolo11SegBranch &a,
                                                  const Yolo11SegBranch &b) {
    return a.height > b.height;
  });
  for (size_t i = 0; i < branches.size() && i < yolo11_seg_config_.strides.size(); ++i) {
    branches[i].stride = yolo11_seg_config_.strides[i];
  }

  std::vector<float> proto_data;
  int proto_h = 0;
  int proto_w = 0;
  int proto_c = 0;
  if (ParseTensorToFloat(output_tensors[proto_index],
                         batch_index,
                         proto_data,
                         proto_h,
                         proto_w,
                         proto_c) != 0) {
    return -1;
  }
  if (proto_c != mask_channels) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "invalid proto channels: %d, expect %d",
                 proto_c,
                 mask_channels);
    return -1;
  }

  const float conf_thres_raw = -std::log(1.0f / score_threshold_ - 1.0f);

  std::vector<Yolo11SegCandidate> candidates;
  for (const auto &branch : branches) {
    const int hw = branch.height * branch.width;
    for (int idx = 0; idx < hw; ++idx) {
      const float *cls_ptr = branch.cls.data() +
                             static_cast<size_t>(idx) * class_num;
      int cls_id = 0;
      float max_cls = cls_ptr[0];
      for (int c = 1; c < class_num; ++c) {
        if (cls_ptr[c] > max_cls) {
          max_cls = cls_ptr[c];
          cls_id = c;
        }
      }
      if (max_cls < conf_thres_raw) {
        continue;
      }

      const float score = hobot::dnn_node::output_parser::Sigmoid(max_cls);
      const float *bbox_ptr = branch.bbox.data() +
                              static_cast<size_t>(idx) * reg_dim * 4;

      float ltrb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      for (int side = 0; side < 4; ++side) {
        auto probs = Softmax(bbox_ptr + side * reg_dim, reg_dim);
        float dist = 0.0f;
        for (int k = 0; k < reg_dim; ++k) {
          dist += probs[static_cast<size_t>(k)] * static_cast<float>(k);
        }
        ltrb[side] = dist;
      }

      const int gh = idx / branch.width;
      const int gw = idx % branch.width;
      float x1 = (static_cast<float>(gw) + 0.5f - ltrb[0]) * branch.stride;
      float y1 = (static_cast<float>(gh) + 0.5f - ltrb[1]) * branch.stride;
      float x2 = (static_cast<float>(gw) + 0.5f + ltrb[2]) * branch.stride;
      float y2 = (static_cast<float>(gh) + 0.5f + ltrb[3]) * branch.stride;

      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      if (cls_id < 0 || cls_id >= static_cast<int>(yolo11_seg_config_.class_names.size())) {
        continue;
      }

      Yolo11SegCandidate cand;
      cand.det = Detection(
          cls_id,
          score,
          Bbox(x1, y1, x2, y2),
          yolo11_seg_config_.class_names[static_cast<size_t>(cls_id)].c_str());

      const float *coeff_ptr = branch.coeff.data() +
                               static_cast<size_t>(idx) * mask_channels;
      cand.coeff.assign(coeff_ptr, coeff_ptr + mask_channels);
      candidates.emplace_back(std::move(cand));
    }
  }

  std::vector<size_t> selected_idx;
  NmsWithIndex(candidates, nms_threshold_, nms_top_k_, selected_idx);

  perception.mask.det_info.clear();
  perception.mask.mask_info.clear();

  int input_h = 0;
  int input_w = 0;
  for (const auto &branch : branches) {
    input_h = std::max(input_h, branch.height * branch.stride);
    input_w = std::max(input_w, branch.width * branch.stride);
  }

  const size_t per_mask_size = static_cast<size_t>(proto_h) * proto_w;
  for (auto idx : selected_idx) {
    if (idx >= candidates.size()) {
      continue;
    }

    const auto &cand = candidates[idx];
    perception.mask.det_info.emplace_back(cand.det);

    // 每个实例输出一张完整proto分辨率的二值mask，便于上层按框裁剪/渲染
    for (int h = 0; h < proto_h; ++h) {
      for (int w = 0; w < proto_w; ++w) {
        const size_t proto_offset =
            (static_cast<size_t>(h) * proto_w + w) * mask_channels;
        float mask_value = 0.0f;
        for (int c = 0; c < mask_channels; ++c) {
          mask_value += cand.coeff[static_cast<size_t>(c)] *
                        proto_data[proto_offset + static_cast<size_t>(c)];
        }
        perception.mask.mask_info.push_back(mask_value > 0.5f ? 1.0f : 0.0f);
      }
    }
  }

  perception.mask.width = proto_w;
  perception.mask.height = proto_h;
  perception.mask.w_base = proto_w > 0
                               ? static_cast<float>(input_w) / static_cast<float>(proto_w)
                               : 1.0f;
  perception.mask.h_base = proto_h > 0
                               ? static_cast<float>(input_h) / static_cast<float>(proto_h)
                               : 1.0f;

  if (!perception.mask.det_info.empty() &&
      perception.mask.mask_info.size() !=
          perception.mask.det_info.size() * per_mask_size) {
    RCLCPP_WARN(rclcpp::get_logger("Yolo11_seg_parser"),
                "mask size mismatch, det=%zu mask=%zu per_mask=%zu",
                perception.mask.det_info.size(),
                perception.mask.mask_info.size(),
                per_mask_size);
  }

  perception.type = Perception::MASK;
  return 0;
}

}  // namespace

int LoadConfig(const rapidjson::Document &document) {
  if (document.HasMember("class_num")) {
    if (InitClassNum(document["class_num"].GetInt()) < 0) {
      return -1;
    }
  }

  if (document.HasMember("cls_names_list")) {
    std::string cls_name_file = document["cls_names_list"].GetString();
    if (InitClassNames(cls_name_file) < 0) {
      return -1;
    }
  }

  if (document.HasMember("strides")) {
    std::vector<int> strides;
    for (size_t i = 0; i < document["strides"].Size(); ++i) {
      strides.push_back(document["strides"][i].GetInt());
    }
    if (InitStrides(strides, 3) < 0) {
      return -1;
    }
  }

  if (document.HasMember("reg")) {
    yolo11_seg_config_.reg_dim = document["reg"].GetInt();
  }

  if (document.HasMember("mask_channels")) {
    yolo11_seg_config_.mask_channels = document["mask_channels"].GetInt();
  }

  if (document.HasMember("score_threshold")) {
    score_threshold_ = document["score_threshold"].GetFloat();
  }

  if (document.HasMember("nms_threshold")) {
    nms_threshold_ = document["nms_threshold"].GetFloat();
  }

  if (document.HasMember("nms_top_k")) {
    nms_top_k_ = document["nms_top_k"].GetInt();
  }

  std::stringstream ss;
  ss << "load yolo11-seg config success, class_num=" << yolo11_seg_config_.class_num
     << ", reg_dim=" << yolo11_seg_config_.reg_dim
     << ", mask_channels=" << yolo11_seg_config_.mask_channels;
  RCLCPP_INFO(rclcpp::get_logger("Yolo11_seg_parser"), "%s", ss.str().c_str());
  return 0;
}

int32_t Parse(const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
              std::shared_ptr<DnnParserResult> &result,
              int batch_index) {
  if (!node_output) {
    return -1;
  }
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception, batch_index);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_seg_parser"),
                 "postprocess return error, code = %d",
                 ret);
    return ret;
  }

  RCLCPP_DEBUG(rclcpp::get_logger("Yolo11_seg_parser"),
               "parse finished, det count: %zu, mask size: %zu",
               result->perception.mask.det_info.size(),
               result->perception.mask.mask_info.size());
  return 0;
}

}  // namespace parser_yolo11_seg
}  // namespace dnn_node
}  // namespace hobot
