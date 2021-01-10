# InferenceHelper_Sample
- Sample project for InferenceHelper (https://github.com/iwatake2222/InferenceHelper )
- Run a simple classification model (MobileNetv2) using several deep leraning frameworks

![Class Diagram](00_doc/class_diagram.png) 

## How to build sample application
### Requirements
- OpenCV 4.x

### Common 
- Get source code
	```sh
	git clone https://github.com/iwatake2222/InferenceHelper_Sample
	cd InferenceHelper_Sample

	git submodule update --init --recursive
	cd InferenceHelper/ThirdParty/tensorflow
	chmod +x tensorflow/lite/tools/make/download_dependencies.sh
	tensorflow/lite/tools/make/download_dependencies.sh
	```

- Download prebuilt libraries
	- Download prebuilt libraries (ThirdParty.zip) from https://github.com/iwatake2222/InferenceHelper/releases/ 
	- Extract it to `InferenceHelper/ThirdParty/`
- Download models
	- Download models (resource.zip) from https://github.com/iwatake2222/InferenceHelper_Sample/releases/ 
	- Extract it to `resource/`

### Windows (Visual Studio)
- Configure and Generate a new project using cmake-gui for Visual Studio 2017 64-bit
	- `Where is the source code` : path-to-InferenceHelper_Sample/pj_cls_mobilenet_v2
	- `Where to build the binaries` : path-to-build	(any)
- Open `main.sln`
- Set `main` project as a startup project, then build and run!

**Note**
When you use Tensorflow Lite in Visual Studio, use `Release` or `RelWithDebInfo` . If you use `Debug` , you will get exception error while running.

### Linux (PC Ubuntu, Raspberry Pi, Jetson Nano, etc.)
```sh
cd pj_cls_mobilenet_v2
mkdir build && cd build
cmake ..
make
./main
```

### Linux (Cross compile for armv7 and aarch64)
```
sudo apt install g++-arm-linux-gnueabi g++-arm-linux-gnueabihf g++-aarch64-linux-gnu

export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
cmake .. -DBUILD_SYSTEM=aarch64

export CC=arm-linux-gnueabi-gcc
export CXX=arm-linux-gnueabi-g++
cmake .. -DBUILD_SYSTEM=armv7
```

You need to link appropreate OpenCV.


### Options (Select Deep Leraning framework)
- Choose one of the following options.
	- *Note* : InferenceHelper itself supports multiple frameworks (i.e. you can set `on` for several frameworks). However, in this sample project the selected framework is used to `create` InferenceHelper instance for the sake of ease. 

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

