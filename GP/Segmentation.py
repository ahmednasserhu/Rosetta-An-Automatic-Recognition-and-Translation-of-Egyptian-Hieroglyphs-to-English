import torch, detectron2
import os
import cv2


from datetime import datetime

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

#Load/Save a Checkpoint
from detectron2.checkpoint import DetectionCheckpointer

#BUILD MODEL
from detectron2.modeling import build_model

# Takes image path and read it
def imageRead(img_path):
    return cv2.imread(img_path)


# Copy the File to google drive and chnage the Path
# Trained Model: https://drive.google.com/file/d/1BnFD0kZXCFqGeO3SC-ZCUvQfD69vwkPL/view?usp=share_link

# Returns Bounding boxes
def predictor(image):
    ARCHITECTURE = "mask_rcnn_R_101_FPN_3x"
    CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE_PATH))
    # Change the path of the file to your  model.pth file after mounting google Drive
    cfg.MODEL.WEIGHTS = 'D:/GP/models/Copy of model_final (1).pth'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    model = build_model(cfg)
    predictor = DefaultPredictor(cfg)
    metadata={}
    outputs = predictor(image)
    boxes = outputs["instances"].pred_boxes.to('cpu')
    return boxes
# Returns updated Bounding boxes
def contour_detection(image, boxes):
    # Perform contour detection using OpenCV
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list of contours that do not overlap with any of the bounding boxes
    undetected_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:
            x, y, w, h = cv2.boundingRect(contour)
            overlaps = False
            for box in boxes:
                x1, y1, x2, y2 = box
                if x >= x1 and y >= y1 and x + w <= x2 and y + h <= y2:
                    overlaps = True
                    break
            if not overlaps:
                undetected_contours.append(contour)

    # Thresholds to control shown boxes
    min_ratio_threshold = 0.14
    max_ratio_threshold = 2

    # Calculate the average area of the detected boxes
    total_area = sum([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
    avg_area = total_area / len(boxes)

    # Set the minimum area threshold as a fraction of the average area
    min_area_ratio = min_ratio_threshold  # You can adjust this parameter as needed
    min_area = avg_area * min_area_ratio

    # Set the maximum area threshold as a fraction of the average area
    max_area_ratio = max_ratio_threshold  # You can adjust this parameter as needed
    max_area = avg_area * max_area_ratio

    # Keep only the bounding boxes that have an area greater than the minimum area threshold
    boxes = [box for box in boxes if ((box[2] - box[0]) * (box[3] - box[1])) > min_area]

    # Remove the bounding boxes that encompass the whole input image
    image_area = image.shape[0] * image.shape[1]
    boxes = [box for box in boxes if ((box[2] - box[0]) * (box[3] - box[1])) < max_area * image_area]

    # Remove the bounding boxes that are completely inside another bounding box
    new_boxes = []
    for i, box in enumerate(boxes):
        is_inside = False
        for j, other_box in enumerate(boxes):
            if i != j and other_box[0] <= box[0] and other_box[1] <= box[1] and other_box[2] >= box[2] and other_box[
                3] >= box[3]:
                is_inside = True
                break
        if not is_inside:
            new_boxes.append(box)
    boxes = new_boxes

    # Remove the bounding boxes that encompass the whole input image
    image_area = image.shape[0] * image.shape[1]
    boxes = [box for box in boxes if ((box[2] - box[0]) * (box[3] - box[1])) < max_area * image_area]

    # Remove the bounding boxes that are mostly inside another bounding box
    new_boxes = []
    for i, box in enumerate(boxes):
        is_inside = False
        for j, other_box in enumerate(boxes):
            if i != j and box[0] >= other_box[0] and box[1] >= other_box[1] and box[2] <= other_box[2] and box[3] <= \
                    other_box[3]:
                overlap_area = (min(box[2], other_box[2]) - max(box[0], other_box[0])) * (
                            min(box[3], other_box[3]) - max(box[1], other_box[1]))
                if overlap_area / ((box[2] - box[0]) * (box[3] - box[1])) >= 0.7:
                    is_inside = True
                    break
        if not is_inside:
            new_boxes.append(box)

    return new_boxes

# Returns Bounding boxes sorted from left/right => top/bottom
def sortBoundingBoxes(boxes):
    list_of_tuplesBoxes = [tuple(lst) for lst in boxes]
    list_of_tuplesBoxes.sort(key=lambda x: (x[0]))
    list_of_tuplesBoxes.sort(key=lambda x: (x[1]))
    return list_of_tuplesBoxes

def cropBoundingBoxImage(image,boxes):
    # Iterate through bounding boxes
    imageList=[]

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        # Define ROI using bounding box coordinates
        roi = image[ymin:ymax, xmin:xmax]

        # TODO: save or the cropped bounding box image
        imageList.append(roi)

    return imageList


def OCR_Segmentation(imagePath):
    image = imageRead(imagePath)
    boxes = sortBoundingBoxes(contour_detection(image, predictor(image)))
    return cropBoundingBoxImage(image,boxes)

#  Testing final segmentation function
img_list = ["D:/GP/cutimg/x1cropped.jpg", "D:/GP/cutimg/x2cropped.jpg"]
for x,img in enumerate(img_list):

    directory = 'D:/GP/images/'+str(x)
    if not os.path.exists(directory):
        os.makedirs(directory)
    imagesList = OCR_Segmentation(img)
    for i, image in enumerate(imagesList):
        cv2.imwrite(os.path.join(directory, "x" + str(i) + "cropped.jpg"), image)
# To show the images in the Image List


