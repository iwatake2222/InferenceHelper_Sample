/* Copyright 2021 iwatake2222

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
#include "classification_engine.h"

/*** Macro ***/
#define TAG "ClassificationEngine"
#define PRINT(...)   COMMON_HELPER_PRINT(TAG, __VA_ARGS__)
#define PRINT_E(...) COMMON_HELPER_PRINT_E(TAG, __VA_ARGS__)

/* Model parameters */
#if defined(INFERENCE_HELPER_ENABLE_OPENCV)
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define MODEL_NAME  "mobilenetv2-1.0.onnx"
#define HAS_BACKGOUND false
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "mobilenetv20_output_flatten0_reshape0"
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
#if 1
#define MODEL_NAME  "mobilenet_v2_1.0_224.tflite"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#else
#define MODEL_NAME  "mobilenet_v2_1.0_224_quant.tflite"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "output"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
#define MODEL_NAME  "mobilenet_v2_1.0_224_quant_edgetpu.tflite"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "output"
#elif defined(INFERENCE_HELPER_ENABLE_TENSORRT)
#include "inference_helper_tensorrt.h"
#define MODEL_NAME  "mobilenet_v2_1.0_224.onnx"
//#define MODEL_NAME   "mobilenet_v2_1.0_224.trt"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#elif defined(INFERENCE_HELPER_ENABLE_NCNN)
#define MODEL_NAME  "mobilenet_v2_1.0_224.param"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#elif defined(INFERENCE_HELPER_ENABLE_MNN)
#define MODEL_NAME  "mobilenet_v2_1.0_224.mnn"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#elif defined(INFERENCE_HELPER_ENABLE_SNPE)
#define MODEL_NAME  "mobilenet_v2_1.0_224.dlc"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input:0"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Softmax:0"
#elif defined(INFERENCE_HELPER_ENABLE_ARMNN)
#if 1
#define MODEL_NAME  "mobilenet_v2_1.0_224.tflite"
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#else
#define MODEL_NAME  "mobilenetv2-1.0.onnx"
#define HAS_BACKGOUND false
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "data"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "mobilenetv20_output_flatten0_reshape0"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_NNABLA) || defined(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define MODEL_NAME  "mobilenet_v2_1.0_224.nnp"
#define HAS_BACKGOUND true
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#elif defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME) || defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
#if defined(ANDROID) || defined(__ANDROID__)
#define MODEL_NAME  "mobilenet_v2_1.0_224.all.ort"
#else
#define MODEL_NAME  "mobilenet_v2_1.0_224.onnx"
#endif
#define HAS_BACKGOUND true
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Reshape_1"
#elif defined(INFERENCE_HELPER_ENABLE_LIBTORCH) || defined(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
#define MODEL_NAME  "mobilenet_v2_traced.pt"
#define HAS_BACKGOUND false
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "dummy"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "dummy"
#elif defined(INFERENCE_HELPER_ENABLE_TENSORFLOW) || defined(INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU)
#define MODEL_NAME  "mobilenet_v2/"
#define HAS_BACKGOUND false
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "serving_default_input_1:0"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "StatefulPartitionedCall:0"
#else
#define MODEL_NAME  "error"
#endif

#define LABEL_NAME  "imagenet_labels.txt"


/*** Function ***/
int32_t ClassificationEngine::Initialize(const std::string& work_dir, const int32_t num_threads)
{
    /* Set model information */
    std::string model_filename = work_dir + "/model/" + MODEL_NAME;
    std::string label_filename = work_dir + "/model/" + LABEL_NAME;

    /* Set input tensor info */
    input_tensor_info_list_.clear();
    InputTensorInfo input_tensor_info(INPUT_NAME, TENSORTYPE, IS_NCHW);
    input_tensor_info.tensor_dims = INPUT_DIMS;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.normalize.mean[0] = 0.485f;   /* https://github.com/onnx/models/tree/master/vision/classification/mobilenet#preprocessing */
    input_tensor_info.normalize.mean[1] = 0.456f;
    input_tensor_info.normalize.mean[2] = 0.406f;
    input_tensor_info.normalize.norm[0] = 0.229f;
    input_tensor_info.normalize.norm[1] = 0.224f;
    input_tensor_info.normalize.norm[2] = 0.225f;
    input_tensor_info_list_.push_back(input_tensor_info);

    /* Set output tensor info */
    output_tensor_info_list_.clear();
    output_tensor_info_list_.push_back(OutputTensorInfo(OUTPUT_NAME, TENSORTYPE));

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


    /* read label */
    if (ReadLabel(label_filename, label_list_) != kRetOk) {
        return kRetErr;
    }

    return kRetOk;
}

int32_t ClassificationEngine::Finalize()
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    inference_helper_.reset();
    return kRetOk;
}


int32_t ClassificationEngine::Process(const cv::Mat& original_mat, Result& result)
{
    if (!inference_helper_) {
        PRINT_E("Inference helper is not created\n");
        return kRetErr;
    }

    /*** PreProcess ***/
    const auto& t_pre_process0 = std::chrono::steady_clock::now();
    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
#if 1
    /** Use image data as input **/
    /* do resize and color conversion here because some inference engine doesn't support these operations */
    int32_t crop_x = 0;
    int32_t crop_y = 0;
    int32_t crop_w = original_mat.cols;
    int32_t crop_h = original_mat.rows;
    cv::Mat img_src = cv::Mat::zeros(input_tensor_info.GetHeight(), input_tensor_info.GetWidth(), CV_8UC3);
    CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeStretch);
    //CommonHelper::CropResizeCvt(original_mat, img_src, crop_x, crop_y, crop_w, crop_h, IS_RGB, CommonHelper::kCropTypeCut);
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
#else
    /** Use blob data as input (img->blob conversion is done by InferenceHelper::PreProcessByOpenCV)**/
    cv::Mat img_src;
    input_tensor_info.data = original_mat.data;
    input_tensor_info.data_type = InputTensorInfo::kDataTypeImage;
    input_tensor_info.image_info.width = original_mat.cols;
    input_tensor_info.image_info.height = original_mat.rows;
    input_tensor_info.image_info.channel = original_mat.channels();
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = original_mat.cols;
    input_tensor_info.image_info.crop_height = original_mat.rows;
#ifdef CV_COLOR_IS_RGB
    input_tensor_info.image_info.is_bgr = IS_RGB;
    input_tensor_info.image_info.swap_color = false;
#else
    input_tensor_info.image_info.is_bgr = IS_RGB;
    input_tensor_info.image_info.swap_color = true;
#endif
#if 0
    InferenceHelper::PreProcessByOpenCV(input_tensor_info, false, img_src);
    input_tensor_info.data_type = InputTensorInfo::kDataTypeBlobNhwc;
#else
    InferenceHelper::PreProcessByOpenCV(input_tensor_info, true, img_src);
    input_tensor_info.data_type = InputTensorInfo::kDataTypeBlobNchw;
#endif
    input_tensor_info.data = img_src.data;
#endif

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
    std::vector<float> output_score_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());

    /* Find the max score */
    int32_t max_index = (int32_t)(std::max_element(output_score_list.begin(), output_score_list.end()) - output_score_list.begin());
    auto max_score = *std::max_element(output_score_list.begin(), output_score_list.end());
    PRINT("Result = %s (%d) (%.3f)\n", label_list_[max_index].c_str(), max_index, max_score);
    const auto& t_post_process1 = std::chrono::steady_clock::now();

    /* Return the results */
    result.class_id = max_index;
    result.class_name = label_list_[max_index];
    result.score = max_score;
    result.time_pre_process = static_cast<std::chrono::duration<double>>(t_pre_process1 - t_pre_process0).count() * 1000.0;
    result.time_inference = static_cast<std::chrono::duration<double>>(t_inference1 - t_inference0).count() * 1000.0;
    result.time_post_process = static_cast<std::chrono::duration<double>>(t_post_process1 - t_post_process0).count() * 1000.0;;

    return kRetOk;
}


int32_t ClassificationEngine::ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        PRINT_E("Failed to read %s\n", filename.c_str());
        return kRetErr;
    }
    label_list.clear();
    if (HAS_BACKGOUND) {
        label_list.push_back("background");
    }
    std::string str;
    while (getline(ifs, str)) {
        label_list.push_back(str);
    }
    return kRetOk;
}
