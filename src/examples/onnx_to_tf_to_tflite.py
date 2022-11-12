import onnx
import tensorflow as tf
import numpy as np
from onnx_tf.backend import prepare

onnx_model = onnx.load_model("model/backup/model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("model/backup/model.tf")

model = tf.saved_model.load("model/backup/model.tf")
model.trainable = False
input_tensor = tf.random.uniform([1, 3, 80, 80])
# todo requires named input!
#out = model(**{'input': input_tensor})
#print(out.shape)

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("model/backup/model.tf")
tflite_model = converter.convert()

# Save the model
with open("model/backup/model.tflite", 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="model/backup/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)