import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

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

def predict(image_paths, classifier_model):
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
labels = {0: 'A1', 1: 'A40', 2: 'Aa15', 3: 'D12', 4: 'D2', 5: 'D21', 6: 'D28',
          7: 'D35', 8: 'D36', 9: 'D4', 10: 'D46', 11: 'D47', 12: 'D54', 13: 'D55',
          14: 'D58', 15: 'D67', 16: 'E23', 17: 'E34', 18: 'E9', 19: 'F34', 20: 'G1',
          21: 'G17', 22: 'G25', 23: 'G35', 24: 'G36', 25: 'G39', 26: 'G4', 27: 'G40',
          28: 'G43', 29: 'G5', 30: 'G7', 31: 'H6', 32: 'I10', 33: 'I9', 34: 'M17', 35:'M17a',
          36: 'M18', 37: 'M23', 38: 'M35', 39: 'N1', 40: 'N11', 41: 'N14', 42: 'N18',
          43: 'N25', 44: 'N29', 45: 'N30', 46: 'N31', 47: 'N33', 48: 'N35', 49: 'N37', 50: 'N5',
          51: 'O1', 52: 'O28', 53: 'O34', 54: 'O4', 55: 'O49', 56: 'O50', 57: 'P8', 58: 'Q1', 59: 'Q3',
          60: 'R8', 61: 'S29', 62: 'S34', 63: 'U1', 64: 'U15', 65: 'U33', 66: 'U7', 67: 'V13', 68: 'V28',
          69: 'V30', 70: 'V31', 71: 'V31a', 72: 'V4', 73: 'W24', 74: 'W25', 75: 'X1', 76: 'X8', 77: 'Y1',
          78: 'Y5', 79: 'Z1', 80: 'Z11', 81: 'Z2', 82: 'Z3'}

# Example usage
# image_paths = ['‪C:/Users/lenovo/Desktop/Newfolder/x0cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x1cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x2cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x3cropped.jpg', '‪C:/Users/lenovo/Desktop/Newfolder/x4cropped.jpg']
image_paths = [r'D:\GP\images\y1.jpg', r'D:\GP\images\y2.jpg', r'D:\GP\images\y3.jpg', r'D:\GP\images\y4.jpg']

classifier_model = 'D:/GP/models/resnet50V2_fine_tuned.hdf5'
prediction_results = predict(image_paths, classifier_model)
print(prediction_results)
output_string = "-".join(prediction_results)

print(output_string)
