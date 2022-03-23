import numpy as np
import torch
from torchvision import models

model = models.mobilenet_v2(pretrained=True)
model.to('cpu')
model.eval()

dummy_input = torch.randn((1, 3, 224, 224))
torch.onnx.export(model, dummy_input, "mobilenet_v2_op11.onnx", opset_version=11)
torch.onnx.export(model, dummy_input, "mobilenet_v2.onnx", opset_version=9)
## Note: Use opset 9 because OpenCV (in Windows) doesn't support a clip operator with 3 inputs which is exposed by opset 11
##       In opset 9, a clip operator has 2 attributes(min, max) and 1 input

# sm = torch.jit.script(model)
# sm.save("mobilenet_v2_jit.pt")
traced_script_module = torch.jit.trace(model, dummy_input)
traced_script_module.save("mobilenet_v2.jit.pt")
