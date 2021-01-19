import onnxruntime
import cv2
import numpy as np


def process_img(img_path):
    img = cv2.imread(img_path)
    img1 = cv2.resize(img, (304,304))
    image = img[:, :, ::-1].transpose((2,0,1))
    image = image[np.newaxis, :, :, :]/255
    image = np.array(image, dtype=np.float32)
    return img1, image

session = onnxruntime.InferenceSession("./pelee.onnx")
input_name = [input.name for input in session.get_inputs()][0]
output_name = [output.name for output in session.get_outputs()]

print("input name: ", input_name, "output name: ", output_name)

image_path = "/home/dp/Desktop/algorithms/Pelee.Pytorch/imgs/COCO/1_65.jpg"
img, image = process_img(image_path)
imggg = cv2.imread(image_path)
loc, conf =  session.run(output_name, {input_name: image})
loc_data = loc.data
conf_data = conf.data
w, h = imggg.shape[1], imggg.shape[0]
scale = np.array([w, h, w, h])
boxes = loc_data
print(loc_data.shape)