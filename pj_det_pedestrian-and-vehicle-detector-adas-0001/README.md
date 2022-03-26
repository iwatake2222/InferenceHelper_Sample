# pedestrian-and-vehicle-detector-adas-0001 with TensorFlowLite/TensorRT/OpenCV/OpenVINO/MNN/ArmNN/NNabla/ONNXRuntime in C++ on Windows/Linux/Linux(Arm)/Android

![00_doc/demo.jpg](00_doc/demo.jpg)]

## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/InferenceHelper_Sample
2. Build  `pj_det_pedestrian-and-vehicle-detector-adas-0001` project (this directory)


## Tested environments
| Framework                 | Windows (x64)   | Linux (x64)     | Linux (aarch64) | Android (aarch64) |
|---------------------------|-----------------|-----------------|-----------------|-------------------|
| TensorFlow Lite           | [x]             | [x]             | [x]             | [x]               |
| TensorFlow Lite + XNNPACK | [x]             | [x]             | [x]             | [x]               |
| TensorFlow Lite + EdgeTPU | [ ] bad result  | [ ] bad result  | [ ] bad result  | Unsupported       |
| TensorFlow Lite + GPU     | No library      |  No library     | No library      | [x]               |
| TensorFlow Lite + NNAPI   | No library      |  No library     | No library      | [x]               |
| TensorRT                  | [x]             | [ ]             | [x]             | No library        |
| OpenCV(dnn)               | [x]             | [x]             | [x]             | [x]               |
| OpenVINO with OpenCV      | [x]             | [x]             | Unsupported     | Unsupported       |
| ncnn                      | [ ] error       | [ ] error       | No library      | [ ] error         |
| MNN                       | [x]             | [x]             | [x]             | [x]               |
| ~~SNPE~~                  | Unsupported     | Unsupported     | [ ]             | [ ]               |
| Arm NN                    | Unsupported     | [ ]             | [x]             | No library        |
| NNabla                    | [x]             | No library      | No library      | No library        |
| ONNX Runtime              | [x]             | [x]             | [x]             | [x]               |
| LibTorch                  | No model        | No model        | No model        | No model          |
| TensorFlow                | [ ] error       | [ ] error       | No library      | No library        |

## Note
- To run with OpenVINO, enable OpenCV and uncomment the following line
    - `//#define MODEL_NAME  "pedestrian-and-vehicle-detector-adas-0001/model_float32.xml"   /* for OpenVINO */`
- To run on Android, modify `ViewAndroid\app\src\main\cpp\CMakeLists.txt`
    - `set(ImageProcessor_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../../pj_det_pedestrian-and-vehicle-detector-adas-0001/image_processor")`

## Acknowledgements
- https://docs.openvino.ai/latest/omz_models_model_pedestrian_and_vehicle_detector_adas_0001.html
- https://github.com/PINTO0309/PINTO_model_zoo
- Drive Video by Dashcam Roadshow
    - 4K横浜ドライブ 新横浜→みなとみらい→横浜ベイブリッジ
    - https://www.youtube.com/watch?v=JreKvvItrnM
