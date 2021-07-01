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
#ifndef IMAGE_PROCESSOR_H_
#define IMAGE_PROCESSOR_H_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <array>

namespace cv {
    class Mat;
};


typedef struct {
    char     workDir[256];
    int32_t  numThreads;
} INPUT_PARAM;

typedef struct {
    int32_t classId;
    char    label[256];
    double  score;
    double  timePreProcess;   // [msec]
    double  timeInference;    // [msec]
    double  timePostProcess;  // [msec]
} OUTPUT_PARAM;

int32_t ImageProcessor_initialize(const INPUT_PARAM* inputParam);
int32_t ImageProcessor_process(cv::Mat* mat, OUTPUT_PARAM* outputParam);
int32_t ImageProcessor_finalize(void);
int32_t ImageProcessor_command(int32_t cmd);

#endif
