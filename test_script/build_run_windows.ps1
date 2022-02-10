# Run on Visual Studio 2019 Developer PowerShell
# You may need the following command before executing this script
# Set-ExecutionPolicy Unrestricted -Scope Process

Param([string]$FRAMEWORK_NAME = "MNN")

echo "Build for: " "INFERENCE_HELPER_ENABLE_$FRAMEWORK_NAME"

if(Test-Path build){
    del -R build
}
mkdir build
cd build
cmake -DINFERENCE_HELPER_ENABLE_"$FRAMEWORK_NAME"=on ../pj_cls_mobilenet_v2_wo_opencv
MSBuild -m:4 ./main.sln /p:Configuration=Release
if(!($?)) {
    echo "Build failed"
    cd ..
    return -1
}


./Release/main.exe
if(!($?)) {
    echo "Execution error"
    cd ..
    return -1
}


echo "OK"
cd ..
echo "$FRAMEWORK_NAME" >> time_inference_windows.txt
cat build/time_inference.txt >> time_inference_windows.txt

return 0