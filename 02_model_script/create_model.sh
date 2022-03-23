##### On Ubuntu20_model_generation (Client OS)
### Create PyTorch model and ONNX model
python3 create_model_pytorch_mobilenet_v2.py

### Simplify ONNX model
mv mobilenet_v2.onnx temp.onnx
onnxsim temp.onnx mobilenet_v2.onnx
rm temp.onnx

### Craete ONNX Runtime model
python3 -m onnxruntime.tools.convert_onnx_models_to_ort mobilenet_v2.onnx

### Craete ncnn model (copy onnx2ncnn to the current directory)
./onnx2ncnn mobilenet_v2.onnx mobilenet_v2.ncnn.param mobilenet_v2.ncnn.bin

### Craete MNN model (copy MNNConvert to the current directory)
./MNNConvert -f ONNX --modelFile mobilenet_v2.onnx --MNNModel mobilenet_v2.mnn --bizCode biz

### Create OpenVINO model 
python3 /opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py --input_model mobilenet_v2.onnx

### Create TensorFlow and TensorFlow Lite model
python3 create_model_tf_mobilenet_v2.py
edgetpu_compiler mobilenet_v2_quant.tflite

##### Run the following commands in WSL2 (outside a docker)
### Create NNabla model (run the following commands in WSL2 (outside a docker))
docker run --rm -v /mnt/c/iwatake/devel:/devel -e DISPLAY="192.168.1.2:0" -it nnabla/nnabla:py37-v1.20.1
nnabla_cli convert mobilenet_v2.onnx mobilenet_v2.nnp

### Create SNPE model
snpe-onnx-to-dlc --input_network mobilenet_v2.onnx \
                 --output_path mobilenet_v2.dlc
