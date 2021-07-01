#include <jni.h>
#include <string>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "ImageProcessor.h"

//#define WORK_DIR    "/sdcard/resource/"
//#define WORK_DIR    "/mnt/sdcard/resource"
//#define WORK_DIR    "/storage/emulated/0/resource/"
#define WORK_DIR    "/storage/emulated/0/Android/data/com.iwatake.viewandroidinferencehelpersample/files/Documents/resource"

static std::mutex g_mtx;

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidinferencehelpersample_MainActivity_ImageProcessorInitialize(
        JNIEnv* env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    InputParam inputParam;
    snprintf(inputParam.work_dir, sizeof(inputParam.work_dir), WORK_DIR);
    inputParam.num_threads = 4;
    ret = ImageProcessor_initialize(&inputParam);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidinferencehelpersample_MainActivity_ImageProcessorProcess(
        JNIEnv* env,
        jobject, /* this */
        jlong   objMat) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    cv::Mat* mat = (cv::Mat*) objMat;
    OutputParam outputParam;
    ret = ImageProcessor_process(mat, &outputParam);
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidinferencehelpersample_MainActivity_ImageProcessorFinalize(
        JNIEnv* env,
        jobject /* this */) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    ret = ImageProcessor_finalize();
    return ret;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_iwatake_viewandroidinferencehelpersample_MainActivity_ImageProcessorCommand(
        JNIEnv* env,
        jobject, /* this */
        jint cmd) {

    std::lock_guard<std::mutex> lock(g_mtx);
    int ret = 0;
    ret = ImageProcessor_command(cmd);
    return ret;
}

