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
def cropped_images(image2,directory, lst_tuples):
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

    horizontal = cv2.dilate(horizontal, (1,1), iterations=5)
    horizontal = cv2.erode(horizontal, (1,1), iterations=5)

    ## Uncomment to visualize highlighted Horizontal lines
    # cv2.imshow("horizontal",horizontal)
    # cv2.waitKey(0)

    # HoughlinesP function to detect horizontal lines
    hor_lines = cv2.HoughLinesP(horizontal,rho=1,theta=np.pi/180,threshold=350,minLineLength=30,maxLineGap=3)
    #if hor_lines is None:
    #    return None,None
    if hor_lines is None:
        pass
        #print('do nothing')
    else:

        temp_line = []
        for line in hor_lines:
           for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1-5,x2,y2-5])

    # Sorting the list of detected lines by Y1
        hor_lines = sorted(temp_line,key=lambda x: x[1])

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
        i=0
        for x1,y1,x2,y2 in hor_lines:
            if y1 >= lasty1 and y1 <= lasty1 + 10:
                lines_x1.append(x1)
                lines_x2.append(x2)
            else:
               if (i != 0 and len(lines_x1)  !=  0):
                hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
            lasty1 = y1
            lines_x1 = []
            lines_x2 = []
            lines_x1.append(x1)
            lines_x2.append(x2)
            i+=1
        hor.append([min(lines_x1),lasty1,max(lines_x2),lasty1])
    #####################################################################


    # [vertical lines]
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical = cv2.dilate(vertical, (1,1), iterations=8)
    vertical = cv2.erode(vertical, (1,1), iterations=7)

    ######## Preprocessing Vertical Lines ###############
    # cv2.imshow("vertical",vertical)
    # cv2.waitKey(0)
    #####################################################

    # HoughlinesP function to detect vertical lines
    # ver_lines = cv2.HoughLinesP(vertical,rho=1,theta=np.pi/180,threshold=20,minLineLength=20,maxLineGap=2)
    ver_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 300, np.array([]), 20, 2)
    if ver_lines is None:
        return hor,None
    temp_line = []
    for line in ver_lines:
        for x1,y1,x2,y2 in line:
            temp_line.append([x1,y1,x2,y2])

    # Sorting the list of detected lines by X1
    ver_lines = sorted(temp_line,key=lambda x: x[0])

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
    for x1,y1,x2,y2 in ver_lines:
        if x1 >= lastx1 and x1 <= lastx1 + 15 and not (((min(y1,y2)<min(lasty1,lasty2)-20 or min(y1,y2)<min(lasty1,lasty2)+20)) and ((max(y1,y2)<max(lasty1,lasty2)-20 or max(y1,y2)<max(lasty1,lasty2)+20))):
            lines_y1.append(y1)
            lines_y2.append(y2)
            # lasty1 = y1
            # lasty2 = y2
        else:
            if (count != 0 and len(lines_y1) != 0):
                ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])
            lastx1 = x1
            lines_y1 = []
            lines_y2 = []
            lines_y1.append(y1)
            lines_y2.append(y2)
            count += 1
            lasty1 = -11111
            lasty2 = -11111
    ver.append([lastx1,min(lines_y2)-5,lastx1,max(lines_y1)-5])

    if hor_lines is None:   
        return None,ver
       
    #################################################################


    ############ Visualization of Lines After Post Processing ############
    # for x1, y1, x2, y2 in ver:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)

    # for x1, y1, x2, y2 in hor:
    #     cv2.line(img, (x1,y1), (x2,y2), (0, 255, 0), 1)
    
    # cv2.imshow("image",img)
    # cv2.waitKey(0)
    #######################################################################
    
    return hor,ver #result ====>result[0]--->hor               result[1]----->ver
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
    if (direction == "Vertical" or direction == "vertical" ):
        cropped_images = verticalLineAndCroppedColumns(img, x, folder_path)
    else:
        cropped_images = horizontalLineAndCroppedColumns(img, x, folder_path)

    encoded_images = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode() for img in cropped_images]
    return jsonify({'images': encoded_images})





def verticalLineAndCroppedColumns(image,numOfColumns,path):
    image2=image
    result=line_detection(image)
    y1=0
    y2=image.shape[0]
    points=[None] * numOfColumns
    i=int(image.shape[1]/numOfColumns)
    min=image.shape[1]
    for z in range(numOfColumns):
        if(z!=numOfColumns-1):
          min=image.shape[1]
          for y in range(len(result[1])):
            value=abs(i*(z+1)-result[1][y][0])
            if(value<min):
              min=value
              point=[result[1][y][0], y1, result[1][y][2], y2]
              points[z]=point
        else:
            point=[image.shape[1], y1, image.shape[1], y2]
            points[z]=point
    X1=[]
    Y1=[]
    X2=[]
    W=[]
    H=[]
    for x in range(len(points)):
      H.append(image.shape[0])
      Y1.append(points[x][1])
      if(x==0):
        X1.append(0)
        W.append(points[x][2]-0)
        continue
      else:
        X1.append(points[x-1][0])
      if(x==len(points)-1):
        W.append(image.shape[1]-points[x-1][0])
      else:
        W.append(points[x][2]-points[x-1][0])
    lst_tuple = list(zip(X1,Y1,W,H))
    return cropped_images(image2 ,path ,lst_tuple)

 #######################################################################   
def horizontalLineAndCroppedColumns(image,numOfRows,path):

    result=line_detection(image)
  
    x1=0
    x2=image.shape[1]

    points=[None] * numOfRows
    i=int(image.shape[0]/numOfRows)#image.shape[1] is the image width -----9 num of coulmns in image get from user 
    min=image.shape[0]
    for z in range(numOfRows):#forloop for the calculated line 12 lines

        if(z!=numOfRows-1):
            min=image.shape[0]#width 1150
            for y in range(len(result[0])):#result[1]----->ver lines form hough line -       len(result[1])=30 lines from hough lines
               value=abs(i*(z+1)-result[0][y][1])#i*(z+1)==>96*(0+1) 96-1=95   96-105=9   96-198=102
               if(value<min):
                min=value#9           XX2=[70, 198, 342, 469, 612]
                point=[x1, result[0][y][1], x2,result[0][y][3]]# point =[105,0,105,1600]
                points[z]=point#point[0]=point

        else:
            point=[x1, image.shape[0], x2,image.shape[0]]# point =[105,0,105,1600]
            points[z]=point#point[0]=point
    X1=[]
    Y1=[]
    X2=[]
    W=[]
    H=[]
    for x in range(len(points)):
      #H.append(image.shape[0])
      W.append(image.shape[1])
      X1.append(points[x][0])
      if(x==0):
        Y1.append(0)
        #W.append(points[x][2]-0)
        H.append(points[x][3]-0)
        continue
      else:
        Y1.append(points[x-1][1])#[[70, 198, 342, 469] points[x][0]
      if(x==len(points)-1):#[70, 128, 144, 127, 143] 
        #W.append(image.shape[1]-points[x][0])
        #H.append(points[x][3]-0)
        H.append(image.shape[0]-points[x-1][1])
      else:
        #W.append(points[x][2]-points[x-1][0])
        H.append(points[x][3]-points[x-1][1])
        
    lst_tuple = list(zip(X1,Y1,W,H))
    return cropped_images(image ,path, lst_tuple)
"""path='C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\image4'
image=cv2.imread('C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\image4\\image4.jpg')
verticalLineAndCroppedColumns(image,5,path)"""
#path='C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\ROWTEST1'
#image=cv2.imread('C:\\Users\\MANDE\\Desktop\\alaa first trem4\\Pictures\\Pictures\\ROWTEST1\\ROWTEST11.jpg')
#horizontalLineAndCroppedColumns(image,6,path)
# path='C:/Users/lenovo/Desktop/Newfolder'
# image=cv2.imread('C:/Users/lenovo/Desktop/Newfolder/X.jpg')
# verticalLineAndCroppedColumns(image,12,path)

if __name__ == "__main__":
    app.run(host="192.168.1.3", debug=False)