cmake_minimum_required(VERSION 3.0)

set(LibraryName "ImageProcessor")

# Create library
add_library (${LibraryName} ImageProcessor.cpp ImageProcessor.h ClassificationEngine.cpp ClassificationEngine.h)

# For OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${LibraryName} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${LibraryName} ${OpenCV_LIBS})

# Link Common Helper module
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/CommonHelper CommonHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/CommonHelper)
target_link_libraries(${LibraryName} CommonHelper)

# Link InferenceHelper module
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_LIST_DIR}/../../third_party/)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/InferenceHelper InferenceHelper)
target_include_directories(${LibraryName} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../InferenceHelper/InferenceHelper)
target_link_libraries(${LibraryName} InferenceHelper)

# Add definition to select framework automatically. You don't need this in the real product
if(INFERENCE_HELPER_ENABLE_OPENCV)
	add_definitions(-DINFERENCE_HELPER_ENABLE_OPENCV)
endif()
if(INFERENCE_HELPER_ENABLE_TENSORRT)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TENSORRT)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_GPU)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_EDGETPU)
endif()
if(INFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
	add_definitions(-DINFERENCE_HELPER_ENABLE_TFLITE_DELEGATE_XNNPACK)
endif()
if(INFERENCE_HELPER_ENABLE_NCNN)
	add_definitions(-DINFERENCE_HELPER_ENABLE_NCNN)
endif()
if(INFERENCE_HELPER_ENABLE_MNN)
	add_definitions(-DINFERENCE_HELPER_ENABLE_MNN)
endif()
