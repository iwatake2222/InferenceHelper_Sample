/* Copyright 2022 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/*** Include ***/
/* for general */
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <chrono>
#include <fstream>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "common_helper.h"
#include "common_helper_cv.h"
#include "inference_helper.h"
#include "prior_bbox.h"
#include "detection_engine.h"

/*** Macro ***/
#define TAG "DetectionEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(INFERENCE_HELPER_ENABLE_OPENCV)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.onnx"
//#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.xml"   /* for OpenVINO */
//#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/pedestrian-and-vehicle-detector-adas-0001.xml"   /* to output prior_box from OpenVINO model */
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
#if 1
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.tflite"
#define INPUT_NAME  "serving_default_data:0"
#define INPUT_DIMS  { 1, 384, 672, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#else
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_full_integer_quant.tflite"
#define INPUT_NAME  "serving_default_data:0"
#define INPUT_DIMS  { 1, 384, 672, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_full_integer_quant_edgetpu.tflite"
#define INPUT_NAME  "serving_default_data:0"
#define INPUT_DIMS  { 1, 384, 672, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#elif defined(INFERENCE_HELPER_ENABLE_TENSORRT)
#include "inference_helper_tensorrt.h"
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.onnx"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_NCNN)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.ncnn.param"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_MNN)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.mnn"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_SNPE)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.dlc"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_ARMNN)
#if 1
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.tflite"
#define INPUT_NAME  "serving_default_data:0"
#define INPUT_DIMS  { 1, 384, 672, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#else
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.onnx"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_NNABLA) || defined(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.nnp"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#elif defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME) || defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
#if defined(ANDROID) || defined(__ANDROID__)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.all.ort"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#else
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.onnx"
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 384, 672 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME_0 "tf.identity"
#define OUTPUT_NAME_1 "tf.identity_1"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_LIBTORCH) || defined(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
#elif defined(INFERENCE_HELPER_ENABLE_TENSORFLOW) || defined(INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU)
#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/saved_model"
#define INPUT_NAME  "serving_default_data:0"
#define INPUT_DIMS  { 1, 384, 672, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME_0 "StatefulPartitionedCall:0"
#define OUTPUT_NAME_1 "StatefulPartitionedCall:1"
#else
#define MODEL_NAME  "error"
#endif

static const std::vector<std::string> kLabelList{ "Vehicle", "Pedestrian" };
static constexpr int32_t kNumClass = 2;
static constexpr int32_t kIntervalConfidence = (1 + kNumClass);

/*** Function ***/
int32_t DetectionEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    /* Normalize [0, 255] */
    input_tensor_info.normalize.mean[0] = 0.0f;
    input_tensor_info.normalize.mean[1] = 0.0f;
    input_tensor_info.normalize.mean[2] = 0.0f;
    input_tensor_info.normalize.norm[0] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[1] = 1.0f / 255.0f;
    input_tensor_info.normalize.norm[2] = 1.0f / 255.0f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_0, TENSORTYPE));
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME_1, TENSORTYPE));
    //output_tensor_info_list_.push_back(OutputTensorInfo("mbox_conf_flatten", TENSORTYPE));    /* to output prior_box from OpenVINO model */
    //output_tensor_info_list_.push_back(OutputTensorInfo("mbox_loc", TENSORTYPE));
    //output_tensor_info_list_.push_back(OutputTensorInfo("mbox_priorbox", TENSORTYPE));


    /* Create and Initialize Inference Helper */
#if defined(INFERENCE_HELPER_ENABLE_OPENCV)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOpencv));
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLite));
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteXnnpack));
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteGpu));
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteEdgetpu));
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowLiteNnapi));
#elif defined(INFERENCE_HELPER_ENABLE_TENSORRT)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorrt));
    if (inference_helper_) {
        InferenceHelperTensorRt* p = dynamic_cast<InferenceHelperTensorRt*>(inference_helper_.get());
        if (p) {
            p->SetDlaCore(-1);  /* Use GPU */
        }
    }
#elif defined(INFERENCE_HELPER_ENABLE_NCNN)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kNcnn));
#elif defined(INFERENCE_HELPER_ENABLE_MNN)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kMnn));
#elif defined(INFERENCE_HELPER_ENABLE_SNPE)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kSnpe));
#elif defined(INFERENCE_HELPER_ENABLE_ARMNN)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kArmnn));
#elif defined(INFERENCE_HELPER_ENABLE_NNABLA)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kNnabla));
#elif defined(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kNnablaCuda));
#elif defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntime));
#elif defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kOnnxRuntimeCuda));
#elif defined(INFERENCE_HELPER_ENABLE_LIBTORCH)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kLibtorch));
#elif defined(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kLibtorchCuda));
#elif defined(INFERENCE_HELPER_ENABLE_TENSORFLOW)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflow));
#elif defined(INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU)
    inference_helper_.reset(InferenceHelper::Create(InferenceHelper::kTensorflowGpu));
#else
    PRINT_E("Inference Helper type is not selected\n");
#endif
    if (!inference_helper_) {
        return kRetErr;
    }
    if (inference_helper_->SetNumThreads(num_threads) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }
    if (inference_helper_->Initialize(model_filename, input_tensor_info_list_, output_tensor_info_list_) != InferenceHelper::kRetOk) {
        inference_helper_.reset();
        return kRetErr;
    }

    return kRetOk;
}

int32_t DetectionEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    inference_helper_.reset();
    return kRetOk;
}


int32_t DetectionEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeExpand);

    input_tensor_info.data = img_src.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = img_src.cols;
    input_tensor_info.image_info.height = img_src.rows;
    input_tensor_info.image_info.channel = img_src.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = img_src.cols;
    input_tensor_info.image_info.crop_height = img_src.rows;
    input_tensor_info.image_info.is_bgr = IS_RGB;
    input_tensor_info.image_info.swap_color = false;

    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_pre_process1 = std::chrono::steady_clock::now();

    /*** Inference ***/
    const auto& t_inference0 = std::chrono::steady_clock::now();
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }
    const auto& t_inference1 = std::chrono::steady_clock::now();

    /*** PostProcess ***/
    const auto& t_post_process0 = std::chrono::steady_clock::now();
    /* Retrieve the result */
    const std::vector<float> output_bbox_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());
    const std::vector<float> output_confidence_list(output_tensor_info_list_[1].GetDataAsFloat(), output_tensor_info_list_[1].GetDataAsFloat() + output_tensor_info_list_[1].GetElementNum());

    /* to output prior_box from OpenVINO model */
    //const std::vector<float> output_prior_box(output_tensor_info_list_[2].GetDataAsFloat(), output_tensor_info_list_[2].GetDataAsFloat() + 2 * 48512);
    //FILE *fp = fopen("prior_box.txt", "w");
    //for (int32_t i = 0; i < output_prior_box.size(); i++) {
    //    fprintf(fp, "%.8f, ", output_prior_box[i]);
    //    if (i % 4 == 3) {
    //        fprintf(fp, "\n");
    //    }
    //}
    //fclose(fp);

     /* Get boundig box */
    static const size_t kNumPrior = PRIOR_BBOX::BBOX.size() / 4;
    std::vector<BoundingBox> bbox_list;
    for (size_t i = 0; i < kNumPrior; i++) {
        float box_score = output_confidence_list[i * kIntervalConfidence + 0];
        if (box_score >= threshold_box_confidence_) {
            int32_t class_index = (int32_t)(std::max_element(&output_confidence_list[i * kIntervalConfidence + 1], &output_confidence_list[i * kIntervalConfidence + kIntervalConfidence]) - &output_confidence_list[i * kIntervalConfidence + 1]);
            float class_score = *std::max_element(&output_confidence_list[i * kIntervalConfidence + 1], &output_confidence_list[i * kIntervalConfidence + kIntervalConfidence]);
            if (class_score >= threshold_class_confidence_) {
                /* Prior Box: [0.0, 1.0] */
                const float prior_x0 = PRIOR_BBOX::BBOX[i * 4 + 0];
                const float prior_y0 = PRIOR_BBOX::BBOX[i * 4 + 1];
                const float prior_x1 = PRIOR_BBOX::BBOX[i * 4 + 2];
                const float prior_y1 = PRIOR_BBOX::BBOX[i * 4 + 3];
                const float prior_cx = (prior_x0 + prior_x1) / 2.0f;
                const float prior_cy = (prior_y0 + prior_y1) / 2.0f;
                const float prior_w = prior_x1 - prior_x0;
                const float prior_h = prior_y1 - prior_y0;

                /* Detected Box: [0.0, MODEL_SIZE] -> [0.0, 1.0] */
                float box_cx = output_bbox_list[i * 4 + 0];
                float box_cy = output_bbox_list[i * 4 + 1];
                float box_w = output_bbox_list[i * 4 + 2];
                float box_h = output_bbox_list[i * 4 + 3];

                /* Adjust box [0.0, 1.0] */
                /* Reference: */
                /*   https://github.com/openvinotoolkit/openvino/blob/17091476d86cbb98392216fa3d4f0db90914449a/inference-engine/thirdparty/clDNN/src/impls/cpu/detection_output.cpp#L135 */
                /*   https://docs.openvino.ai/latest/openvino_docs_ops_detection_DetectionOutput_1.html */
                /*   https://github.com/openvinotoolkit/open_model_zoo/tree/2021.4.2/models/intel/vehicle-detection-0200 */
                float cx = PRIOR_BBOX::VARIANCE[0] * box_cx * prior_w + prior_cx;
                float cy = PRIOR_BBOX::VARIANCE[1] * box_cy * prior_h + prior_cy;
                float w = std::exp(box_w * PRIOR_BBOX::VARIANCE[2]) * prior_w;
                float h = std::exp(box_h * PRIOR_BBOX::VARIANCE[3]) * prior_h;

                /* Store the detected box */
                auto bbox = BoundingBox{
                    class_index,
                    kLabelList[class_index],
                    class_score,
                    static_cast<int32_t>((cx - w / 2.0) * crop_w),
                    static_cast<int32_t>((cy - h / 2.0) * crop_h),
                    static_cast<int32_t>(w * crop_w),
                    static_cast<int32_t>(h * crop_h)
                };
                bbox_list.push_back(bbox);
            }
        }
    }


    /* Adjust bounding box */
    for (auto& bbox : bbox_list) {
        bbox.x += crop_x;
        bbox.y += crop_y;
    }

    /* NMS */
    std::vector<BoundingBox> bbox_nms_list;
    BoundingBoxUtils::Nms(bbox_list, bbox_nms_list, threshold_nms_iou_);

    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.bbox_list = bbox_nms_list;
    result.crop.x = (std::max)(0, crop_x);
    result.crop.y = (std::max)(0, crop_y);
    result.crop.w = (std::min)(crop_w, original_mat.cols - result.crop.x);
    result.crop.h = (std::min)(crop_h, original_mat.rows - result.crop.y);
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}

