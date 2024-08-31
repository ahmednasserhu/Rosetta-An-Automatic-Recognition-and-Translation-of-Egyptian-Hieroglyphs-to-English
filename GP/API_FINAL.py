import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import glob
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def cropped_images(image2, directory, lst_tuples):
    cropped_images = []
    for i, crop_area in enumerate(lst_tuples, 1):
        x, y = crop_area[0], crop_area[1]
        w, h = crop_area[2], crop_area[3]
        crop_img = image2[y:y + h, x:x + w]
        cv2.imwrite(os.path.join(directory, "x" + str(i) + "cropped.jpg"), crop_img)
        cropped_images.append(crop_img)
    return cropped_images


# Input : Image
# Output : hor,ver
def line_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 1)
    bw = cv2.bitwise_not(bw)
    ## To visualize image after thresholding ##
    # cv2.imshow("bw",bw)
    # cv2.waitKey(0)
    ###########################################
    horizontal = bw.copy()
    vertical = bw.copy()
    img = image.copy()
    # [horizontal lines]
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    horizontal = cv2.dilate(horizontal, (1, 1), iterations=5)
    horizontal = cv2.erode(horizontal, (1, 1), iterations=5)

    ## Uncomment to visualize highlighted Horizontal lines
    # cv2.imshow("horizontal",horizontal)
    # cv2.waitKey(0)

    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal, rho=1, theta=np.pi / 180, threshold=350, minLineLength=30, maxLineGap=3)
    # if hor_lines is None:
    #    return None,None
    if hor_lines is None:
        pass
        # print('do nothing')
    else:

        temp_line = []
        for line in hor_lines:
            for x1, y1, x2, y2 in line:
                temp_line.append([x1, y1 - 5, x2, y2 - 5])

        # Sorting the list of detected lines by Y1
        hor_lines = sorted(temp_line, key=lambda x: x[1])

        ## Uncomment this part to visualize the lines detected on the image ##
        # print(len(hor_lines))
        # for x1, y1, x2, y2 in hor_lines:
        #     cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 1)

        # print(image.shape)
        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        ####################################################################

        ## Selection of best lines from all the horizontal lines detected ##
        lasty1 = -111111
        lines_x1 = []
        lines_x2 = []
        hor = []
        i = 0
        for x1, y1, x2, y2 in hor_lines:
            if y1 >= lasty1 and y1 <= lasty1 + 10:
                lines_x1.append(x1)
                lines_x2.append(x2)
            else:
                if (i != 0 and len(lines_x1) != 0):
                    hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i += 1
        hor.append([min(lines_x1), lasty1, max(lines_x2), lasty1])
    #####################################################################

    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1, 1), iterations=8)
    vertical = cv2.erode(vertical, (1, 1), iterations=7)

    ######## Preprocessing Vertical Lines ###############
    # cv2.imshow("vertical",vertical)
    # cv2.waitKey(0)
    #####################################################

    # HoughlinesP function to detect vertical lines
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi / 180, 300, np.array([]), 20, 2)
    if ver_lines is None:
        return hor, None
    temp_line = []
    for line in ver_lines:
        for x1, y1, x2, y2 in line:
            temp_line.append([x1, y1, x2, y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line, key=lambda x: x[0])

    ## Uncomment this part to visualize the lines detected on the image ##
    # print(len(ver_lines))
    # for x1, y1, x2, y2 in ver_lines:
    #     cv2.line(image, (x1,y1-5), (x2,y2-5), (0, 255, 0), 1)

    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    ####################################################################

    ## Selection of best lines from all the vertical lines detected ##
    lastx1 = -111111
    lines_y1 = []
    lines_y2 = []
    ver = []
    count = 0
    lasty1 = -11111
    lasty2 = -11111
    for x1, y1, x2, y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (
                ((min(y1, y2) < min(lasty1, lasty2) - 20 or min(y1, y2) < min(lasty1, lasty2) + 20)) and (
        (max(y1, y2) < max(lasty1, lasty2) - 20 or max(y1, y2) < max(lasty1, lasty2) + 20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if (count != 0 and len(lines_y1) != 0):
                ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1, min(lines_y2) - 5, lastx1, max(lines_y1) - 5])

    if hor_lines is None:
        return None, ver

    #################################################################

    ############ Visualization of Lines After Post Processing ############
    # for x1, y1, x2, y2 in ver:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # for x1, y1, x2, y2 in hor:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    #######################################################################

    return hor, ver  # result ====>result[0]--->hor               result[1]----->ver
    #######################################################################


import base64


@app.route('/handle_image_and_integer_and_path', methods=['POST'])
def handle_image_and_integer_and_path():
    image = request.files['image'].read()
    x = int(request.form['x'])
    folder_path = request.form['path']
    direction = request.form['direction']
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if (direction == "Vertical" or direction == "vertical"):
        cropped_images = verticalLineAndCroppedColumns(img, x, folder_path)
    else:
        cropped_images = horizontalLineAndCroppedColumns(img, x, folder_path)

    encoded_images = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode() for img in cropped_images]
    return jsonify({'images': encoded_images})


def verticalLineAndCroppedColumns(image, numOfColumns, path):
    image2 = image
    result = line_detection(image)
    y1 = 0
    y2 = image.shape[0]
    points = [None] * numOfColumns
    i = int(image.shape[1] / numOfColumns)
    min = image.shape[1]
    for z in range(numOfColumns):
        if (z != numOfColumns - 1):
            min = image.shape[1]
            for y in range(len(result[1])):
                value = abs(i * (z + 1) - result[1][y][0])
                if (value < min):
                    min = value
                    point = [result[1][y][0], y1, result[1][y][2], y2]
                    points[z] = point
        else:
            point = [image.shape[1], y1, image.shape[1], y2]
            points[z] = point
    X1 = []
    Y1 = []
    X2 = []
    W = []
    H = []
    for x in range(len(points)):
        H.append(image.shape[0])
        Y1.append(points[x][1])
        if (x == 0):
            X1.append(0)
            W.append(points[x][2] - 0)
            continue
        else:
            X1.append(points[x - 1][0])
        if (x == len(points) - 1):
            W.append(image.shape[1] - points[x - 1][0])
        else:
            W.append(points[x][2] - points[x - 1][0])
    lst_tuple = list(zip(X1, Y1, W, H))
    return cropped_images(image2, path, lst_tuple)


#######################################################################
def horizontalLineAndCroppedColumns(image, numOfRows, path):
    result = line_detection(image)

    x1 = 0
    x2 = image.shape[1]

    points = [None] * numOfRows
    i = int(
        image.shape[0] / numOfRows)  # image.shape[1] is the image width -----9 num of coulmns in image get from user
    min = image.shape[0]
    for z in range(numOfRows):  # forloop for the calculated line 12 lines

        if (z != numOfRows - 1):
            min = image.shape[0]  # width 1150
            for y in range(len(result[
                                   0])):  # result[1]----->ver lines form hough line -       len(result[1])=30 lines from hough lines
                value = abs(i * (z + 1) - result[0][y][1])  # i*(z+1)==>96*(0+1) 96-1=95   96-105=9   96-198=102
                if (value < min):
                    min = value  # 9           XX2=[70, 198, 342, 469, 612]
                    point = [x1, result[0][y][1], x2, result[0][y][3]]  # point =[105,0,105,1600]
                    points[z] = point  # point[0]=point

        else:
            point = [x1, image.shape[0], x2, image.shape[0]]  # point =[105,0,105,1600]
            points[z] = point  # point[0]=point
    X1 = []
    Y1 = []
    X2 = []
    W = []
    H = []
    for x in range(len(points)):
        # H.append(image.shape[0])
        W.append(image.shape[1])
        X1.append(points[x][0])
        if (x == 0):
            Y1.append(0)
            # W.append(points[x][2]-0)
            H.append(points[x][3] - 0)
            continue
        else:
            Y1.append(points[x - 1][1])  # [[70, 198, 342, 469] points[x][0]
        if (x == len(points) - 1):  # [70, 128, 144, 127, 143]
            # W.append(image.shape[1]-points[x][0])
            # H.append(points[x][3]-0)
            H.append(image.shape[0] - points[x - 1][1])
        else:
            # W.append(points[x][2]-points[x-1][0])
            H.append(points[x][3] - points[x - 1][1])

    lst_tuple = list(zip(X1, Y1, W, H))
    return cropped_images(image, path, lst_tuple)


"""path='C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\image4'
image=cv2.imread('C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\image4\\image4.jpg')
verticalLineAndCroppedColumns(image,5,path)"""
# path='C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\ROWTEST1'
# image=cv2.imread('C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\ROWTEST1\\ROWTEST11.jpg')
# horizontalLineAndCroppedColumns(image,6,path)
# path='C:/Users/lenovo/Desktop/Newfolder'
# image=cv2.imread('C:/Users/lenovo/Desktop/Newfolder/X.jpg')
# verticalLineAndCroppedColumns(image,12,path)


# import os
# import requests
# from flask import Flask, request
# from flask_cors import CORS
# app = Flask(__name__)
# CORS(app)
# @app.route("/process_images", methods=["POST"])
# def process_images():
#     image_files = request.files.getlist('images[]')
#     print(image_files)
#     save_directory = 'C:/Users/lenovo/Desktop/Newfolder/x'
#
#     # Create a directory to save the images if it doesn't exist
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)
#
#     saved_image_paths = []
#
#     # Iterate over the received files
#     for index, image_file in enumerate(image_files):
#         # Save the image to the server
#         save_path = os.path.join(save_directory, f'image_{index}.jpg')
#         image_file.save(save_path)
#         saved_image_paths.append(save_path)
#
#     # Return a response with the saved image paths
#     return {'message': 'Images saved successfully', 'saved_image_paths': saved_image_paths}
#
# if __name__ == "__main__":
#     app.run()
import numpy as np
import tensorflow as tf
import subprocess
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import svm, metrics, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS


@app.route('/process_images', methods=['POST'])
def process_images():
    if 'images' in request.files:
        images = request.files.getlist('images')
        file_list = os.listdir('D:/GP/houghlineoutput2/')

        # Iterate over the files and delete each image
        for file_name in file_list:
            file_path = os.path.join('D:/GP/houghlineoutput2/', file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
        for i, image in enumerate(images):
            print(image)
            image.save('D:/GP/houghlineoutput2/' "segmen" + str(i) + ".jpg")
        translation=segmentaion()
        print(images)

        #     img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        #
        # # Save the image locally
        #     cv2.imwrite('path/to/save/' + image.filename, img)

        return jsonify(translation)
    else:
        return 'No images found'


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

import shutil
#  Testing final segmentation function
def segmentaion():
    import glob

    folder_path = "D:/GP/houghlineoutput2"
    file_pattern = "segmen*.jpg"

    img_list = glob.glob(folder_path + "/" + file_pattern)
    paths = [[] for i in range(len(img_list))]
    for x,img in enumerate(img_list):
        imagesList=[]
        paths_final=[[]]
        imagesList = OCR_Segmentation(img)
        directory='D:/GP/segmintation/image'+str(x)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # To show the images in the Image List
        for i,image in enumerate(imagesList):
            cv2.imwrite(os.path.join(directory, "x" + str(i) + "cropped.jpg"), image)

            paths[x].append(directory+"/x"+ str(i) + "cropped.jpg")

    print(paths[1])
    classifier_model = 'D:/GP/models/resnet50V2_fine_tuned.hdf5'
    output_string = ""
    for i, path in enumerate(paths):
        prediction_results = predict(paths[i], classifier_model)
        print(prediction_results)
        output_string = output_string + "-".join(prediction_results) + "\n"

    print(output_string)
    file_path = "D:/GP/models/ancient_egypt_dictionary_compiled.xlsx"  # Update with the path to your Excel file
    dictionary = load_dictionary_from_excel(file_path)

    text = output_string
    segmented_words = forward_maximum_segmentation(text, dictionary)

    mapped_words = []
    for word in segmented_words:
        if word in dictionary:
            mapped_words.append(dictionary[word])
        else:
            mapped_words.append(word)

    print(mapped_words)

    rulebased_string = ' '.join(mapped_words)
    output_file = 'D:/GP/test_eg.txt'
    with open(output_file, 'w') as file:
        file.write(rulebased_string)
    command = [
        'onmt_translate',
        '-model', 'D:/GP/models/translation_step_3000.pt',
        '-src', 'D:/GP/test_eg.txt',
        '-output', 'D:/GP/output.txt',
        '-replace_unk',
        '-gpu', '0'
    ]

    # Execute the command using subprocess
    subprocess.run(command, shell=True)


    with open("D:/GP/output.txt", "r") as myfile:
        data = myfile.read().splitlines()
        print(data)
    return data






def predict(image_paths, classifier_model):
    labels = {0: 'A1', 1: 'A40', 2: 'Aa15', 3: 'D12', 4: 'D2', 5: 'D21', 6: 'D28',
              7: 'D35', 8: 'D36', 9: 'D4', 10: 'D46', 11: 'D47', 12: 'D54', 13: 'D55',
              14: 'D58', 15: 'D67', 16: 'E23', 17: 'E34', 18: 'E9', 19: 'F34', 20: 'G1',
              21: 'G17', 22: 'G25', 23: 'G35', 24: 'G36', 25: 'G39', 26: 'G4', 27: 'G40',
              28: 'G43', 29: 'G5', 30: 'G7', 31: 'H6', 32: 'I10', 33: 'I9', 34: 'M17', 35: 'M17a',
              36: 'M18', 37: 'M23', 38: 'M35', 39: 'N1', 40: 'N11', 41: 'N14', 42: 'N18',
              43: 'N25', 44: 'N29', 45: 'N30', 46: 'N31', 47: 'N33', 48: 'N35', 49: 'N37', 50: 'N5',
              51: 'O1', 52: 'O28', 53: 'O34', 54: 'O4', 55: 'O49', 56: 'O50', 57: 'P8', 58: 'Q1', 59: 'Q3',
              60: 'R8', 61: 'S29', 62: 'S34', 63: 'U1', 64: 'U15', 65: 'U33', 66: 'U7', 67: 'V13', 68: 'V28',
              69: 'V30', 70: 'V31', 71: 'V31a', 72: 'V4', 73: 'W24', 74: 'W25', 75: 'X1', 76: 'X8', 77: 'Y1',
              78: 'Y5', 79: 'Z1', 80: 'Z11', 81: 'Z2', 82: 'Z3'}
    # Load the model from the HDF5 file
    model = load_model(classifier_model,compile=False)

    # Initialize an empty list to store the predicted class names
    predicted_class_names = []

    # Iterate through the list of image paths
    for image_path in image_paths:
        # Preprocess the input image
        test_image = image.load_img(image_path, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # Perform the prediction
        predictions = model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()

        # Interpret the prediction results
        class_names = labels
        predicted_class_name = class_names[np.argmax(scores)]
        predicted_class_names.append(predicted_class_name)

    return predicted_class_names


# Example usage
# image_paths = ['‪C:/Users/lenovo/Desktop/Newfolder/x0cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x1cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x2cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x3cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x4cropped.jpg']
# image_paths = [r'C:\Users\lenovo\Desktop\Newfolder\y1.jpeg', r'C:\Users\lenovo\Desktop\Newfolder\y2.jpeg', r'C:\Users\lenovo\Desktop\Newfolder\y3.jpeg', r'C:\Users\lenovo\Desktop\Newfolder\y4.jpeg']





# -*- coding: utf-8 -*-
"""rule based algorthim

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12AXCBeZAFnCIzL9xt0fysfKzFSziWysR
"""


def load_dictionary_from_excel(file_path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_path)

    # Assuming the Excel columns are named "Word" and "Mapping"
    dictionary = {}
    for index, row in df.iterrows():
        word = str(row['Gardiner'])
        mapping = str(row['Transliteration'])
        dictionary[word] = mapping

    return dictionary

def forward_maximum_segmentation(text, dictionary):
    segmented_words = []
    i = 0
    while i < len(text):
        found_word = False
        for j in range(len(text), i, -1):
            # Try to find the longest possible word starting from position i
            word = text[i:j]
            if word in dictionary:
                segmented_words.append(dictionary[word])
                i = j
                found_word = True
                break

        # If no word is found, treat the current character as a separate word
        if not found_word:
            segmented_words.append(text[i])
            i += 1

    return segmented_words

# Example usage





# Define the command

if __name__ == "__main__":
    app.run()