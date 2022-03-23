import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=None, alpha=1.0, include_top=True, weights='imagenet',
    input_tensor=None, pooling=None, classes=1000,
    classifier_activation='softmax'
)
model.compile()

### Create SavedModel
model.save('mobilenet_v2/saved_model/')

### Create tflite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("mobilenet_v2.tflite", "wb").write(tflite_model)

### Create tflite model (Post-training quantization)
ds = tfds.load(
    name="coco/2017",
    with_info=False,
    split="validation",
    data_dir="./tfds",
    download=True
)

shape = (224, 224) 
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def representative_dataset_gen():
    for data in ds.take(1000):
        image = data['image'].numpy()
        preprocessed_data = tf.image.resize(image, (shape[0], shape[1]))
        preprocessed_data = preprocessed_data / 255.
        preprocessed_data = (preprocessed_data - mean) / std
        preprocessed_data = preprocessed_data[np.newaxis,:,:,:]
        yield [preprocessed_data]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# converter.experimental_new_quantizer = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
open("mobilenet_v2_quant.tflite", "wb").write(tflite_model)
