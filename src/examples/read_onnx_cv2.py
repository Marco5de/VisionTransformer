"""
Minimal example on how to do ONNX inference with OpenCV - works the same in C++
"""
import cv2
import numpy as np

ONNX_MODEL_PATH = "model/backup/model.onnx"

t = np.zeros([1, 3, 80, 80])

net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
net.setInput(t)
out = net.forward()
print(t.shape, out.shape)