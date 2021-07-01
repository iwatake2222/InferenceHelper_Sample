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
#ifndef CLASSIFICATION_ENGINE_
#define CLASSIFICATION_ENGINE_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <memory>

/* for OpenCV */
#include <opencv2/opencv.hpp>

/* for My modules */
#include "inference_helper.h"


class ClassificationEngine {
public:
    enum {
        RET_OK = 0,
        RET_ERR = -1,
    };

    typedef struct RESULT_ {
        int32_t     labelIndex;
        std::string labelName;
        float       score;
        double      timePreProcess;		// [msec]
        double      timeInference;		// [msec]
        double      timePostProcess;	// [msec]
        RESULT_() : timePreProcess(0), timeInference(0), timePostProcess(0)
        {}
    } RESULT;

private:
    static constexpr bool WITH_BACKGROUND = true;
    
public:
    ClassificationEngine() {}
    ~ClassificationEngine() {}
    int32_t initialize(const std::string& workDir, const int32_t numThreads);
    int32_t finalize(void);
    int32_t invoke(const cv::Mat& originalMat, RESULT& result);

private:
    int32_t readLabel(const std::string& filename, std::vector<std::string>& labelList);

private:
    std::unique_ptr<InferenceHelper> m_inferenceHelper;
    std::vector<InputTensorInfo> m_inputTensorList;
    std::vector<OutputTensorInfo> m_outputTensorList;
    std::vector<std::string> m_labelList;
};

#endif
