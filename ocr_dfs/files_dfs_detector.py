import sys
import numpy as np

# Ruta del repositorio yolov5: https://github.com/ultralytics/yolov5 
#url_yolov5_home = "../yolov5"
#sys.path.insert(0, url_yolov5_home)

import argparse
import time
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox

# Ruta del mejor modelo evaluado
weights = "models/files_dfs_detector/v1.0.pt"

import pytesseract

def get_text(img):
    if img is None:
        return None
    config = ("-l spa --oem 1 --psm 6 ")
    text_data = None
    text_data = (pytesseract.image_to_string(img, config=config))
    return text_data

def dfs_normalice_boxes_output(boxes):
    result = {}
    result["boxes"] = []
    conf_list = []
    for key in boxes.keys():
        for box in boxes[key]:
            box["type"] = key
            box["class"] = "fichero"
            result["boxes"].append(box)
            conf_list.append(box["conf"])
    result["num_boxes"] = len(conf_list)
    if len(conf_list) > 0:
        result["mean_score"] = float(np.average(conf_list))
        result["min_score"] = min(conf_list)
        result["max_score"] = max(conf_list)
        result["boxes"] = sorted(result["boxes"],key=lambda el:el["conf"],reverse=True)
    else:
        
        result["mean_score"] = 0.0
        result["min_score"] = 0.0
        result["max_score"] = 0.0
        
    return result

    
class FilesDFSDetector:
    """
        This class loads yolov5 model and allows to get boxes and conf to every file detected
        0.4
    """
    conf_thres = 0.15
    iou_thres = 0.45
    classes = None
    augment = True
    agnostic_nms = True
    mode = "image"
    save_dir = ""
    device = "cpu"
    img_size = 640
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    def __init__(self, weights_path, text_dfs_analyzer = None):
        self.model = attempt_load(weights_path, map_location=self.device) # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = check_img_size(self.img_size, s=self.stride)  # check img_size # check img_size
        if self.half:
            self.model.half()  # to FP16
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        self.text_dfs_analyzer = text_dfs_analyzer

    def get_predictions(self,img):
        img, im0s = self.resize_img(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        #t1 = time_synchronized()
        pred1 = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred1, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        #t2 = time_synchronized()

        predictions = {0:[], 1:[]}
        # Process detections
        for i, det in enumerate(pred): 
            im0 = im0s

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    predictions[int(cls)].append({"conf": float(conf), "box":[int(el) for el in xyxy]})

        return dfs_normalice_boxes_output(predictions)
    
    
    
    def resize_img(self, img0): 
        # BGR
        assert img0 is not None, 'Image is Null '

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0
    
    
    def get_dfs_angle_attributes(self, img):

        is_dfs_file = 0
        img_zero,  predictions_zero = self.predictions_zero_angle(img)
        img_90 , predictions_90 = self.predictions_90_angle(img)
        img_menos90,  predictions_menos90 = self.predictions_menos90_angle(img)
        img_180 , predictions_180 = self.predictions_180_angle(img)
        #print("L")
        #print([predictions_zero["max_score"], predictions_90["max_score"], predictions_menos90["max_score"], predictions_180["max_score"]])
        is_dfs_file_prob = float(max([predictions_zero["max_score"], predictions_90["max_score"], predictions_menos90["max_score"], predictions_180["max_score"]]))

        is_dfs_file = 1 if is_dfs_file_prob >= 0.4 else 0

        if is_dfs_file == 0:
            return None
        first_img_zero = None 
        first_img_90 = None
        first_img_menos90 = None
        first_img_180 = None

        if len(predictions_zero["boxes"]) > 0:
            first_img_zero = get_image_from_box(img_zero ,predictions_zero["boxes"][0]["box"])
        if len(predictions_90["boxes"]) > 0:
            first_img_90 = get_image_from_box(img_90 ,predictions_90["boxes"][0]["box"])
        if len(predictions_menos90["boxes"]) > 0:
            first_img_menos90 = get_image_from_box(img_menos90 ,predictions_menos90["boxes"][0]["box"])
        if len(predictions_180["boxes"]) > 0:
            first_img_180 = get_image_from_box(img_180 ,predictions_180["boxes"][0]["box"])
        
        array_imgs = [first_img_zero, first_img_90, first_img_menos90, first_img_180]
        array_ocr_lectures = [get_text(img) for img in array_imgs]
        array_ocr_punctuations = [(self.text_dfs_analyzer.evaluate([txt_img])[1][0] if txt_img is not None else 0.0) for txt_img in array_ocr_lectures]
        max_arg = int(np.argmax(array_ocr_punctuations))

        if max_arg == 0:
            return is_dfs_file, is_dfs_file_prob, 0, img_zero, predictions_zero 
        if max_arg == 1:
            return is_dfs_file, is_dfs_file_prob, 90, img_90, predictions_90 
        if max_arg == 2:
            return is_dfs_file, is_dfs_file_prob, -90, img_menos90, predictions_menos90 
        if max_arg == 3:
            return is_dfs_file, is_dfs_file_prob, 180, img_180, predictions_180
        return None


    def predictions_zero_angle(self, img):
        img = img
        return img, self.get_predictions(img)
    def predictions_90_angle(self, img):
        img = cv2.rotate(img,  cv2.ROTATE_90_CLOCKWISE)
        return img, self.get_predictions(img)
    def predictions_menos90_angle(self, img):
        img = cv2.rotate(img,  cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img, self.get_predictions(img)
    def predictions_180_angle(self, img):
        img = cv2.rotate(img,  cv2.ROTATE_180)
        return img, self.get_predictions(img)
    
# Utils part

def get_image_from_box(image,box, PADDING=0.02):
    padd_int = int(round(min(image.shape[:2])*PADDING))
    y1 = max(box[0]-padd_int,0)
    x1 = max(box[1]-padd_int,0)
    y2 = min(box[2]+padd_int,image.shape[1])
    x2 = min(box[3]+padd_int,image.shape[0])
    return image[x1:x2,y1:y2]







# Utils IMG