import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
#------ converting image to grayscale image 
image=cv2.imread('Five.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(gray,cmap="gray")
plt.show()

#but before detecting the edges we have to make the images blur inorder to avoid
#detecting the noises
#so here we will be using gausian blur function
blur=cv2.GaussianBlur(gray,(11,11),0)#here the first parameter is the input image 
plt.imshow(blur,cmap='gray')
plt.show()

#now here we will use canny edge detection algorithm to find the edges
canny=cv2.Canny(blur,30,150,3) #first  parameter is the previous image which we have blurred
#so that we won't capture the noise
plt.imshow(canny,cmap='gray')
plt.show()


# we need to thicken the edges 
dilated=cv2.dilate(canny,(1,1),iterations=2)
plt.imshow(dilated,cmap='gray')
plt.show()

#countours calculation
(cnt,heirachy)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb,cnt,-1,(0,255),2)
# Create a new directory to save the output images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Extract the contours
for i, contour in enumerate(cnt):
    # Create a bounding box around the contour
    x, y, w, h = cv2.boundingRect(contour)
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[y:y+h, x:x+w]
    
    # Save the cropped image
    cv2.imwrite(os.path.join(output_dir, f'output_image_{i}.jpg'), cropped_image)