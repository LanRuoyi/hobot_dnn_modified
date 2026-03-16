// dnn_node/src/util/output_parser/detection/ptq_yolo11_output_parser.cpp
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
#include "dnn_node/util/output_parser/detection/ptq_yolo11_output_parser.h"

#include <arm_neon.h>

#include <iostream>
#include <queue>
#include <fstream>

#include "dnn_node/util/output_parser/detection/nms.h"
#include "dnn_node/util/output_parser/utils.h"
#include "rapidjson/document.h"
#include "rclcpp/rclcpp.hpp"

namespace hobot {
namespace dnn_node {
namespace parser_yolo11 {

/**
 * YOLO11配置结构体
 */
struct PTQYolo11Config {
  std::vector<int> strides;
  int reg_dim;  // DFL回归维度，默认为16
  int class_num;
  std::vector<std::string> class_names;

  std::string Str() {
    std::stringstream ss;
    ss << "strides: ";
    for (const auto &stride : strides) {
      ss << stride << " ";
    }
    ss << "; class_num: " << class_num;
    ss << "; reg_dim: " << reg_dim;
    return ss.str();
  }
};

PTQYolo11Config default_ptq_yolo11_config = {
    {8, 16, 32},  // strides
    16,           // reg_dim
    3,            // class_num
    {"crack", "pothole","repair"}
};

PTQYolo11Config yolo11_config_ = default_ptq_yolo11_config;
float score_threshold_ = 0.4;
float nms_threshold_ = 0.5;
int nms_top_k_ = 5000;

int InitClassNum(const int &class_num) {
  if(class_num > 0){
    yolo11_config_.class_num = class_num;
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                 "class_num = %d is not allowed, only support class_num > 0",
                 class_num);
    return -1;
  }
  return 0;
}

int InitClassNames(const std::string &cls_name_file) {
  std::ifstream fi(cls_name_file);
  if (fi) {
    yolo11_config_.class_names.clear();
    std::string line;
    while (std::getline(fi, line)) {
      yolo11_config_.class_names.push_back(line);
    }
    int size = yolo11_config_.class_names.size();
    if(size != yolo11_config_.class_num){
      RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                 "class_names length %d is not equal to class_num %d",
                 size, yolo11_config_.class_num);
      return -1;
    }
  } else {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                 "can not open cls name file: %s",
                 cls_name_file.c_str());
    return -1;
  }
  return 0;
}

int InitStrides(const std::vector<int> &strides, const int &model_output_count){
  int size = strides.size();
  if(size != model_output_count){
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                "strides size %d is not equal to model_output_count %d",
                size, model_output_count);
    return -1;
  }
  yolo11_config_.strides.clear();
  for (size_t i = 0; i < strides.size(); i++){
    yolo11_config_.strides.push_back(strides[i]);
  }
  return 0;
}

int LoadConfig(const rapidjson::Document &document) {
  if (document.HasMember("class_num")){
    int class_num = document["class_num"].GetInt();
    if (InitClassNum(class_num) < 0) {
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
    for(size_t i = 0; i < document["strides"].Size(); i++){
      strides.push_back(document["strides"][i].GetInt());
    }
    if (InitStrides(strides, 3) < 0){
      return -1;
    }
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
  return 0;
}

/**
 * @brief 计算softmax
 */
static inline void softmax(const float* input, float* output, int length) {
  float max_val = input[0];
  for (int i = 1; i < length; ++i) {
    if (input[i] > max_val) max_val = input[i];
  }
  
  float sum = 0.0f;
  for (int i = 0; i < length; ++i) {
    output[i] = std::exp(input[i] - max_val);
    sum += output[i];
  }
  
  for (int i = 0; i < length; ++i) {
    output[i] /= sum;
  }
}

int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception);

/**
 * @brief 解析单个输出tensor
 */
static void ParseTensor(std::shared_ptr<DNNTensor> cls_tensor, 
                        std::shared_ptr<DNNTensor> bbox_tensor,
                        int layer,
                        std::vector<Detection> &dets) {
  // Flush memory
  hbSysFlushMem(&(cls_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  hbSysFlushMem(&(bbox_tensor->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int CLASSES_NUM = yolo11_config_.class_num;
  float CONF_THRES_RAW = -std::log(1.0f / score_threshold_ - 1.0f);
  int REG = yolo11_config_.reg_dim;
  int stride = yolo11_config_.strides[layer];

  std::vector<float> class_pred(yolo11_config_.class_num, 0.0);

  //  int *shape = tensor->data_shape.d;
  int height, width;
  auto ret =
      hobot::dnn_node::output_parser::get_tensor_hw(cls_tensor, &height, &width);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                 "get_tensor_hw failed");
  }
  
  // Get pointers
  float* cls_raw = reinterpret_cast<float*>(cls_tensor->sysMem[0].virAddr);
  float* bbox_raw = reinterpret_cast<float*>(bbox_tensor->sysMem[0].virAddr);

  // Process each grid cell
  for (int gh = 0; gh < height; gh++) {
      for (int gw = 0; gw < width; gw++) {
          float* cur_cls = cls_raw;
          float* cur_bbox = bbox_raw;

          // Find max class score
          int cls_id = 0;
          for (int c = 1; c < CLASSES_NUM; c++) {
              if (cur_cls[c] > cur_cls[cls_id]) {
                  cls_id = c;
              }
          }

          // Skip if score is too low
          if (cur_cls[cls_id] < CONF_THRES_RAW) {
              cls_raw += CLASSES_NUM;
              bbox_raw += REG * 4;
              continue;
          }

          // Compute score
          float score = 1.0f / (1.0f + std::exp(-cur_cls[cls_id]));

          // DFL decode (softmax + weighted sum)
          float ltrb[4] = {0};
          for (int i = 0; i < 4; i++) {
              float sum = 0.0f;
              for (int j = 0; j < REG; j++) {
                  int idx = REG * i + j;
                  float dfl = std::exp(cur_bbox[idx]);
                  ltrb[i] += dfl * j;
                  sum += dfl;
              }
              ltrb[i] /= sum;
          }

          // Skip invalid boxes
          if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
              cls_raw += CLASSES_NUM;
              bbox_raw += REG * 4;
              continue;
          }

          // Convert to xyxy
          float x1 = (gw + 0.5f - ltrb[0]) * stride;
          float y1 = (gh + 0.5f - ltrb[1]) * stride;
          float x2 = (gw + 0.5f + ltrb[2]) * stride;
          float y2 = (gh + 0.5f + ltrb[3]) * stride;

          // Store result
          // bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
          // scores[cls_id].push_back(score);
          Bbox bbox(x1, y1, x2, y2);
          dets.emplace_back(
              static_cast<int>(cls_id),
              score,
              bbox,
              yolo11_config_.class_names[static_cast<int>(cls_id)].c_str());

          cls_raw += CLASSES_NUM;
          bbox_raw += REG * 4;
      }
  }
}



int32_t Parse(
    const std::shared_ptr<hobot::dnn_node::DnnNodeOutput> &node_output,
    std::shared_ptr<DnnParserResult> &result) {
  if (!result) {
    result = std::make_shared<DnnParserResult>();
  }

  int ret = PostProcess(node_output->output_tensors, result->perception);
  if (ret != 0) {
    RCLCPP_ERROR(rclcpp::get_logger("Yolo11_detection_parser"),
                "postprocess return error, code = %d", ret);
  }

  std::stringstream ss;
  ss << "Yolo11_detection_parser parse finished, detect count: "
     << result->perception.det.size();
  RCLCPP_DEBUG(rclcpp::get_logger("Yolo11_detection_parser"), "%s", ss.str().c_str());
  return ret;
}

/**
 * @brief 后处理主函数
 */
int PostProcess(std::vector<std::shared_ptr<DNNTensor>> &output_tensors,
                Perception &perception) {
  perception.type = Perception::DET;
  std::vector<Detection> dets;

  // 解析3个检测头 (cls0, reg0, cls1, reg1, cls2, reg2)
  for (size_t i = 0; i < 3; ++i) {
    ParseTensor(output_tensors[i * 2], output_tensors[i * 2 + 1], static_cast<int>(i), dets);
  }
  yolo5_nms(dets, nms_threshold_, nms_top_k_, perception.det, false);
  return 0;
}

}  // namespace parser_yolo11
}  // namespace dnn_node
}  // namespace hobot