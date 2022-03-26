# YOLOX with TensorFlowLite/TensorRT/OpenCV/OpenVINO/ncnn/MNN/ArmNN/ONNXRuntime/TensorFlow in C++ on Windows/Linux/Linux(Arm)/Android

![00_doc/demo.jpg](00_doc/demo.jpg)]

## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/InferenceHelper_Sample
2. Build `pj_det_yolox` project (this directory)


## Tested environments
| Framework                 | Windows (x64)   | Linux (x64)     | Linux (aarch64) | Android (aarch64) |
|---------------------------|-----------------|-----------------|-----------------|-------------------|
| TensorFlow Lite           | [x]             | [x]             | [x]             | [x]               |
| TensorFlow Lite + XNNPACK | [x]             | [x]             | [x]             | [x]               |
| TensorFlow Lite + EdgeTPU | [x] bad result  | [ ]             | [x] bad result  | Unsupported       |
| TensorFlow Lite + GPU     | No library      |  No library     | No library      | [x]               |
| TensorFlow Lite + NNAPI   | No library      |  No library     | No library      | [x]               |
| TensorRT                  | [x]             | [ ]             | [x]             | No library        |
| OpenCV(dnn)               | [x]             | [x]             | [ ]             | [x]               |
| OpenVINO with OpenCV      | [x]             | [x]             | Unsupported     | Unsupported       |
| ncnn                      | [x]             | [x]             | No library      | [x]               |
| MNN                       | [x]             | [x]             | [x]             | [x]               |
| ~~SNPE~~                  | Unsupported     | Unsupported     | [ ]             | [ ]               |
| Arm NN                    | Unsupported     | [ ]             | [x]             | No library        |
| NNabla                    | [ ] no model    | No library      | No library      | No library        |
| ONNX Runtime              | [x]             | [x]             | [x]             | [ ] error         |
| LibTorch                  | No model        | No model        | No model        | No model          |
| TensorFlow                | [x]             | [ ]             | No library      | No library        |

## Note
- To run with OpenVINO, enable OpenCV and uncomment the following line
    - `//#define MODEL_NAME  "yolox_nano_480x640/model_float32.xml"   /* for OpenVINO */`
- To run on Android, modify `ViewAndroid\app\src\main\cpp\CMakeLists.txt`
    - `set(ImageProcessor_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../pj_det_yolox/image_processor")`

## Acknowledgements
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/PINTO0309/PINTO_model_zoo
- https://github.com/Tencent/ncnn