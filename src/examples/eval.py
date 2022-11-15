import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.utils.data
from torchvision.transforms import transforms
from tqdm import tqdm

from src.lib.dataset.SignDataset import SignDataset


def ocv_onnx_eval(model_path, data_loader):
    net = cv2.dnn.readNetFromONNX(model_path)

    train_acc = 0.0
    for batch_idx, (input, label) in tqdm(enumerate(data_loader), total=len(data_loader)):
        input = input.numpy()
        net.setInput(input)
        pred = net.forward()

        pred = torch.tensor(pred)
        cls = pred.argmax(dim=1, keepdim=True)
        train_acc += cls.eq(label.view_as(cls)).sum().item()

    print(f"Accuracy on set - {train_acc / len(data_loader.dataset)}")

def tf_eval(model_path, data_loader):
    # Todo - named inputs required
    pass

def tf_lite_eval(model_path, data_loader):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path="model/backup/model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    print("Input shape from model = ", input_shape)

    train_acc = 0.0
    length = 5000
    for batch_idx, (input, label) in enumerate(tqdm(train_loader, total=len(train_loader))):
        input = input.float().numpy()
        interpreter.set_tensor(input_details[0]['index'], input)
        interpreter.invoke()

        # get_tensor() returns a copy of the tensor data
        # use tensor() in order to get a pointer to the tensor
        pred = interpreter.get_tensor(output_details[0]['index'])

        pred = torch.tensor(pred)
        cls = pred.argmax(dim=1, keepdim=True)
        train_acc += cls.eq(label.view_as(cls)).sum().item()
        # todo - takes too long
        if batch_idx == length:
            break

    print(f"Accuracy on set - {train_acc / length}")



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dset = SignDataset("data/Schilder", train=True, transform=transform, uniform=False)
    train_loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)

    #ocv_onnx_eval("model/backup/model.onnx", train_loader)
    tf_lite_eval("model/backup/model.tflite", train_loader)
