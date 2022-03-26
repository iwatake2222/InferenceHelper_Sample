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
#include <memory>
#include <algorithm>
#include <chrono>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

/*** Macro ***/
#define IMAGE_NAME   RESOURCE_DIR"/parrot.jpg"
#define WORK_DIR     RESOURCE_DIR
#define LOOP_NUM_FOR_TIME_MEASUREMENT 20


/*** Function ***/
static int32_t DL_Initialize(const std::string& work_dir, const int32_t num_threads);
static int32_t DL_Finalize();
static int32_t DL_Process(uint8_t* resized_img);

int32_t main()
{
    /* Load and resize image */
    std::unique_ptr<uint8_t[]> img_src;
    // std::unique_ptr<uint8_t[]> img_resized = std::make_unique<uint8_t[]>(224 * 224 * 3);
    std::unique_ptr<uint8_t[]> img_resized(new uint8_t[224 * 224 * 3]);
    int32_t w, h, c;
    img_src.reset(stbi_load(IMAGE_NAME, &w, &h, &c, 0));
    stbir_resize_uint8(img_src.get(), w, h, 0, img_resized.get(), 224, 224, 0, c);
    stbi_write_jpg("img_reszed.jpg", 224, 224, c, img_resized.get(), 95);


    if (DL_Initialize(WORK_DIR, 4) != 0) {
        return -1;
    }
    if (DL_Process(img_resized.get()) != 0) {
        return -1;
    }
    if (DL_Finalize() != 0) {
        return -1;
    }

    return 0;
}



/*** Deep Learning stuffs ***/
/* for My modules */
#include "inference_helper.h"
static constexpr int32_t kRetOk = 0;
static constexpr int32_t kRetErr = -1;

/* Model parameters */
#if defined(INFERENCE_HELPER_ENABLE_OPENCV)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.onnx"
//#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.xml"    /* for OpenVINO */
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU) || defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_NNAPI)
#if 1
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.tflite"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  true
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#else
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2_quant.tflite"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  true
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2_quant_edgetpu.tflite"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  true
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeUint8
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#elif defined(INFERENCE_HELPER_ENABLE_TENSORRT)
#include "inference_helper_tensorrt.h"
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.onnx"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#elif defined(INFERENCE_HELPER_ENABLE_NCNN)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.ncnn.param"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#elif defined(INFERENCE_HELPER_ENABLE_MNN)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.mnn"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#elif defined(INFERENCE_HELPER_ENABLE_SNPE)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2_1.0_224.dlc"
#define HAS_BACKGOUND true
#define HAS_SOFTMAX  false
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define INPUT_NAME  "input:0"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "MobilenetV2/Predictions/Softmax:0"
#elif defined(INFERENCE_HELPER_ENABLE_ARMNN)
#if 1
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.tflite"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  true
#define INPUT_NAME  "input_1"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "Identity"
#else
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.onnx"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_NNABLA) || defined(INFERENCE_HELPER_ENABLE_NNABLA_CUDA)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.nnp"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#elif defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME) || defined(INFERENCE_HELPER_ENABLE_ONNX_RUNTIME_CUDA)
#if defined(ANDROID) || defined(__ANDROID__)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2_op11.all.ort"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "536"
#else
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.onnx"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "input.1"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "466"
#endif
#elif defined(INFERENCE_HELPER_ENABLE_LIBTORCH) || defined(INFERENCE_HELPER_ENABLE_LIBTORCH_CUDA)
#define MODEL_NAME  "mobilenet_v2/mobilenet_v2.jit.pt"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "dummy"
#define INPUT_DIMS  { 1, 3, 224, 224 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     true
#define IS_RGB      true
#define OUTPUT_NAME "dummy"
#elif defined(INFERENCE_HELPER_ENABLE_TENSORFLOW) || defined(INFERENCE_HELPER_ENABLE_TENSORFLOW_GPU)
#define MODEL_NAME  "mobilenet_v2/saved_model"
#define HAS_BACKGOUND false
#define HAS_SOFTMAX  false
#define INPUT_NAME  "serving_default_input_1:0"
#define INPUT_DIMS  { 1, 224, 224, 3 }
#define TENSORTYPE  TensorInfo::kTensorTypeFp32
#define IS_NCHW     false
#define IS_RGB      true
#define OUTPUT_NAME "StatefulPartitionedCall:0"
#else
#define MODEL_NAME  "error"
#endif

#define LABEL_NAME  "mobilenet_v2/imagenet_labels.txt"


static std::unique_ptr<InferenceHelper> inference_helper_;
static std::vector<InputTensorInfo> input_tensor_info_list_;
static std::vector<OutputTensorInfo> output_tensor_info_list_;
static std::vector<std::string> label_list_;

static int32_t DL_ReadLabel(const std::string& filename, std::vector<std::string>& label_list)
{
    std::ifstream ifs(filename);
    if (ifs.fail()) {
        printf("Failed to read %s\n", filename.c_str());
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


static int32_t DL_Initialize(const std::string& work_dir, const int32_t num_threads)
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
    input_tensor_info.image_info.width = 224;
    input_tensor_info.image_info.height = 224;
    input_tensor_info.image_info.channel = 3;
    input_tensor_info.image_info.crop_x = 0;
    input_tensor_info.image_info.crop_y = 0;
    input_tensor_info.image_info.crop_width = 224;
    input_tensor_info.image_info.crop_height = 224;
    input_tensor_info.image_info.is_bgr = false;
    input_tensor_info.image_info.swap_color = false;
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
    if (DL_ReadLabel(label_filename, label_list_) != kRetOk) {
        return kRetErr;
    }

    return kRetOk;
}

static int32_t DL_Finalize()
{
    if (!inference_helper_) {
        printf("Inference helper is not created\n");
        return kRetErr;
    }
    inference_helper_->Finalize();
    inference_helper_.reset();
    return kRetOk;
}

static inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = static_cast<int32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

static float SoftMaxFast(const float* src, float* dst, int32_t length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator{ 0 };

    for (int32_t i = 0; i < length; ++i) {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int32_t i = 0; i < length; ++i) {
        dst[i] /= denominator;
    }

    return 0;
}

static int32_t DL_Process(uint8_t* resized_img)
{
    if (!inference_helper_) {
        printf("Inference helper is not created\n");
        return kRetErr;
    }

    InputTensorInfo& input_tensor_info = input_tensor_info_list_[0];
    input_tensor_info.data = resized_img;


    if (inference_helper_->PreProcess(input_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }

    /*** Inference ***/
    if (inference_helper_->Process(output_tensor_info_list_) != InferenceHelper::kRetOk) {
        return kRetErr;
    }

    /*** PostProcess ***/
    /* Retrieve the result */
    std::vector<float> output_score_raw_list(output_tensor_info_list_[0].GetDataAsFloat(), output_tensor_info_list_[0].GetDataAsFloat() + output_tensor_info_list_[0].GetElementNum());

    /* Find the max score */
    std::vector<float> output_score_list = output_score_raw_list;
    if (!HAS_SOFTMAX) {
        SoftMaxFast(output_score_raw_list.data(), output_score_list.data(), static_cast<int32_t>(output_score_list.size()));
    }

    int32_t max_index = (int32_t)(std::max_element(output_score_list.begin(), output_score_list.end()) - output_score_list.begin());
    auto max_score = *std::max_element(output_score_list.begin(), output_score_list.end());
    printf("Result = %s (%d) (%.3f)\n", label_list_[max_index].c_str(), max_index, max_score);


    /*** Measure Inference time ***/
    const auto& time_inference_0 = std::chrono::steady_clock::now();
    for (int32_t i = 0; i < LOOP_NUM_FOR_TIME_MEASUREMENT; i++) {
        inference_helper_->Process(output_tensor_info_list_);
    }
    const auto& time_inference_1 = std::chrono::steady_clock::now();
    double time_inference = (time_inference_1 - time_inference_0).count() / 1000000.0;
    time_inference /= LOOP_NUM_FOR_TIME_MEASUREMENT;
    std::ofstream os("time_inference.txt", std::ios::out);
    os << time_inference << " [ms]" << std::endl;
    os.close();
    printf("Inference Time = %lf [ms]\n", time_inference);

    const int32_t kTrueAnswer = HAS_BACKGOUND ? 89 : 88;
    if (max_index == kTrueAnswer) {
        return kRetOk;
    } else {
        return kRetErr;
    }
}

