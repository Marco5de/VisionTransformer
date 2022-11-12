import cv2
import numpy as np

model_path = "model/backup/model.onnx"

t = np.zeros([1, 3, 80, 80])

net = cv2.dnn.readNetFromONNX(model_path)
net.setInput(t)
out = net.forward()
print(t.shape, out.shape)