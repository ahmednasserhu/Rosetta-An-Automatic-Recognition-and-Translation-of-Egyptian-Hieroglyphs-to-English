




# Replace with the URL of your ngrok domain
import requests
import cv2
api_url = "http://192.168.1.3:5000/handle_image_and_integer_and_path"
path='D:/GP/cut img'
x=6
# file_path = 'C:/Users/lenovo/Desktop/X.jpg'

file_path = 'D:/GP/images/X.jpg'
image = cv2.imread(file_path)

# Encode the image as a binary string
_, image_encoded = cv2.imencode('.jpg', image)
image_bytes = image_encoded.tobytes()

# Send the image file, file path, and integer variable to the Flask API
response = requests.post(api_url,
                         data={'x': x, 'path': path },
                         files={'image': ('X.jpg', image_bytes, 'image/jpeg')})