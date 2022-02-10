# Run on Visual Studio 2019 Developer PowerShell
# You may need the following command before executing this script
# Set-ExecutionPolicy Unrestricted -Scope Process

Param(
    [string]$FRAMEWORK_NAME = "MNN",
    [switch]$BUILD_ONLY
)
echo "[iwatake] Build for: INFERENCE_HELPER_ENABLE_$FRAMEWORK_NAME"

echo "[iwatake][$FRAMEWORK_NAME] Build Start"
if(Test-Path build) {
    del -R build
}
mkdir build
cd build
cmake -DINFERENCE_HELPER_ENABLE_"$FRAMEWORK_NAME"=on ../pj_cls_mobilenet_v2_wo_opencv
MSBuild -m:4 ./main.sln /p:Configuration=Release
if(!($?)) {
    echo "[iwatake][$FRAMEWORK_NAME] Build error"
    cd ..
    exit -1
}
echo "[iwatake][$FRAMEWORK_NAME] Build End"

if($BUILD_ONLY) {
    cd ..
    exit 0
}


echo "[iwatake][$FRAMEWORK_NAME] Execution Start"
./Release/main.exe
if(!($?)) {
    echo "[iwatake][$FRAMEWORK_NAME] Execution error"
    cd ..
    exit -1
}
echo "[iwatake][$FRAMEWORK_NAME] Execution End"

echo "[iwatake][$FRAMEWORK_NAME] OK"
cd ..
echo "$FRAMEWORK_NAME" >> time_inference_windows.txt
cat build/time_inference.txt >> time_inference_windows.txt

exit  0
