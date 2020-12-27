# InferenceHelper_Sample
- Sample project for InferenceHelper (https://github.com/iwatake2222/InferenceHelper )
- Run a simple classification model (MobileNetv2) using several deep leraning frameworks:
	- TensorFlow Lite
	- TensorFlow Lite with delegate (GPU, XNNPACK, EdgeTPU)
	- TensorRT
	- OpenCV(dnn)
	- ncnn
	- MNN

- todo: class diagram comes here

## Tested Environment
- Windows 10 (Visual Studio 2017 x64)
- Linux (Xubuntu 18.04 x64)
- Linux (Jetson Xavier NX)

## How to build sample application
### Common 
- Get source code
	```sh
	git clone https://github.com/iwatake2222/play_with_tflite.git
	cd play_with_tflite

	git submodule init
	git submodule update
	cd third_party/tensorflow
	chmod +x tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/lite/tools/make/download_dependencies.sh
	```

- Download prebuilt libraries and models
	- Download prebuilt libraries (third_party.zip) and models (resource.zip) from https://github.com/iwatake2222/InferenceHelper_Sample/releases/ 
	- Extract them to `third_party` and `resource`

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-play_with_tflite/pj_tflite_cls_mobilenet_v2	(for example)
	- `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

**Note**
When you use Tensorflow Lite in Visual Studio, use `Release` or `RelWithDebInfo` . If you use 'Debug', you will get exception error while running.

### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```sh
cd pj_cls_mobilenet_v2
mkdir build && cd build
cmake ..
make
./main
```

### Options (Select Deep Leraning framework)
Choose one of the following options.

```sh
cmake .. \
-DINFERENCE_HELPER_ENABLE_OPENCV=off \
-DINFERENCE_HELPER_ENABLE_TFLITE=off \
-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK=off \
-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU=off \
-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU=off \
-DINFERENCE_HELPER_ENABLE_TENSORRT=off \
-DINFERENCE_HELPER_ENABLE_NCNN=off \
-DINFERENCE_HELPER_ENABLE_MNN=on
```

### Additional commands for Tensorflow Lite + EdgeTPU
```sh
cp libedgetpu.so.1.0 libedgetpu.so.1
sudo LD_LIBRARY_PATH=./ ./main
```

# License
- InferenceHelper_Sample
- https://github.com/iwatake2222/InferenceHelper_Sample
- Copyright 2020 iwatake2222
- Licensed under the Apache License, Version 2.0

# Acknowledgements
- This project utilizes OSS (Open Source Software)
	- [NOTICE.md](NOTICE.md)

