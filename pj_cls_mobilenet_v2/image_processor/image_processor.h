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
    char     work_dir[256];
    int32_t  num_threads;
} InputParam;

typedef struct {
    int32_t class_id;
    char    label[256];
    double  score;
    double  time_pre_process;   // [msec]
    double  time_inference;    // [msec]
    double  time_post_process;  // [msec]
} OutputParam;

int32_t ImageProcessor_Initialize(const InputParam* inputParam);
int32_t ImageProcessor_Process(cv::Mat* mat, OutputParam* outputParam);
int32_t ImageProcessor_Finalize(void);
int32_t ImageProcessor_Command(int32_t cmd);

#endif
