"""
Example script based on https://github.com/sithu31296/PyTorch-ONNX-TFLite
Showing conversion from onnx -> tf -> tflite
As utils provide torch -> onnx - conversion for all formats available and working
"""
import onnx
import tensorflow as tf
import numpy as np
from onnx_tf.backend import prepare

ONNX_MODEL_PATH = "model/backup/model.onnx"
TF_MODEL_PATH ="model/backup/model.tf"
TFLITE_MODEL_PATH = "model/backup/model.tflite"


onnx_model = onnx.load_model(ONNX_MODEL_PATH)
tf_rep = prepare(onnx_model)
tf_rep.export_graph(TFLITE_MODEL_PATH)

model = tf.saved_model.load(TF_MODEL_PATH)
model.trainable = False
input_tensor = tf.random.uniform([1, 3, 80, 80])
# todo requires named input!
#out = model(**{'input': input_tensor})
#print(out.shape)

converter = tf.lite.TFLiteConverter.from_saved_model(TF_MODEL_PATH)
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]["index"], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]["index"])