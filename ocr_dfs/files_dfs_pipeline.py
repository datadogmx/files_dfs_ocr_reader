from ocr_dfs.files_dfs_detector import get_image_from_box, FilesDFSDetector, get_text
from ocr_dfs.text_dfs_analyzer import TextDFSEvaluator
from ocr_dfs.clean_image_pipeline import PipelineImg
import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_best_punctutated_transformation( imgs, names, text_evaluator:TextDFSEvaluator):
    """
        This function predict the best transformation for ocr and returns the best img, its text, its description and original ocr
    """
    ocr_imgs = [get_text(img) for img in imgs]
    ocr_puntcutations = text_evaluator.evaluate(ocr_imgs)[1]
    img_selected = int(np.argmax(ocr_puntcutations))
    return imgs[img_selected], ocr_imgs[img_selected], names[img_selected], ocr_imgs[0]

def get_best_lecture( img, boxes, text_evaluator:TextDFSEvaluator, debug=False):
    """
        This function execute img clean transformations and  it evaluates its text punctuation.

    """
    
    images = [get_image_from_box(img,box["box"]) for box in boxes]
    pipelines = [PipelineImg.do_full_pipeline_yolov(sub_img, debug=debug, figsize=(15,15)) for sub_img in images]
    
    names=["original"]+["transformacion_{}".format(fl) for fl in range(len(pipelines[0][0]))]
    
    return [ get_best_punctutated_transformation([images[indx]]+pipelines[indx][0], names=names, text_evaluator = text_evaluator) for indx in range(len(images))]
    get_image_from_box(image,box["box"])

def execute_all_steps( path, detector:FilesDFSDetector, text_evaluator:TextDFSEvaluator, debug=False):
    """
        params: 
                detector: FilesDFSDetector
                text_evaluator: TextDFSEvaluator

        This function executes all steps needed to read a dfs file:
            1.- Rotation Detection
            2.- File DFS Detector
            3.- Pre-processing IMGs
            4.- Returns *img* tagging with metadata next metadata: {
                is_dfs_file: flag used to identify if the img contains a dfs file
                is_dfs_file_prob: metric used to know the probability of the prediction
                angle: Angle in degrees that the algorithm predict the image is rotated
                boxes: Array of dfs files's positions on the img
                num_boxes: how many dfs files were detected
                lectures: Array of the lectures and caracteristics of the best transformation
            }
    """
    #try:
    img = cv2.imread(path)
    response = {"is_dfs_file":0,"is_dfs_file_prob":0.0,"angle":0,"boxes":None}
    result = detector.get_dfs_angle_attributes(img)
    
    if result is not None:
        is_dfs_file, is_dfs_file_prob, angle, img_rotated, boxes = result

        response["is_dfs_file"] = is_dfs_file
        response["is_dfs_file_prob"] = is_dfs_file_prob
        response["angle"] = angle
        response["boxes"] = boxes["boxes"]
        response["num_boxes"] = boxes["num_boxes"]
        response["lectures"] = []
        best_lectures = get_best_lecture(img_rotated, boxes["boxes"], debug=debug, text_evaluator=text_evaluator) 
        for lec in best_lectures:
            response["lectures"].append({
            "best_lecture_text":lec[1],
            "name_transformation":lec[2],
            "normal_lecture_text":lec[3],
            })
    return [path, response ]
    print(".",end="", sep="")
    #except:
    print("-",end="", sep="")
    return [path, None]