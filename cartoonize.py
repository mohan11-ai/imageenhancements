
    
    
    #a = np.double(img)
    #b = a + 15
    #img = np.uint8(b)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #(thresh,img) = cv2.threshold(img, 59, 255, cv2.THRESH_BINARY)

def cartoon(img):
    import cv2
    import numpy as np
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #print(gamaValue)
    for gamma in [3.6]:
    
	
	# Apply gamma correction.
        img = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
        
# denoising of image saving it into dst imag
       # dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 15, 15)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        #(thresh, blackAndWhiteImage) = cv2.threshold(img, 69, 255, cv2.THRESH_BINARY)
         #img = cv2.Canny(img,200,300)    
   
    return img
  
    
def hist(img):
    import cv2
    import numpy as np
   
        
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #contrast = img_grey.std()
    img = cv2.equalizeHist(img_grey)
    #(thresh, img) = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    return img

def blurring(img):
    import cv2
    import numpy as np


    img = cv2.resize(img, (0, 0), None, .25, .25)

    gaussianBlurKernel = np.array(([[1, 2, 1], [2, 4, 2], [1, 2, 1]]), np.float32)/9
    sharpenKernel = np.array(([[0, -1, 0], [-1, 9, -1], [0, -1, 0]]), np.float32)/9
    meanBlurKernel = np.ones((3, 3), np.float32)/9

    gaussianBlur = cv2.filter2D(src=img, kernel=gaussianBlurKernel, ddepth=-1)
    meanBlur = cv2.filter2D(src=img, kernel=meanBlurKernel, ddepth=-1)
    sharpen = cv2.filter2D(src=img, kernel=sharpenKernel, ddepth=-1)

    img= np.concatenate((img, gaussianBlur, meanBlur, sharpen), axis=1)
    
    return img
    
def sharpening(img):
    import cv2
    import numpy as np   
    kernel3 = np.array([[0, -1,  0],
	            [-1,  5, -1],

	            [0, -1,  0]])

    img = cv2.filter2D(src=img, ddepth=-1, kernel=kernel3)
    
    return img
    
def bilateral(img):

    import cv2  
    import numpy as np  

  

    img = cv2.bilateralFilter(img,9,50,50)
    return img 
    
    
def blackwhite(img):

    import cv2
  

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
    (thresh, img) = cv2.threshold(img, 69, 255, cv2.THRESH_BINARY)
    
    return img
    
    
def bright(img, value=70):
    import cv2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
    
    
def RGBcolour(img):


    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    return img
def HSVcolour(img):


    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    
    return img    
    
def HLScolour(img):


    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    
    
    return img
    
    
def invert(img):
    import numpy as np
   
    img = np.invert(img)
    
    return img
    
    
def denoise(img):

    import cv2
  
# denoising of image saving it into dst image
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    
    return img
    
    
    
def morph(img):

    # Python program to demonstrate erosion and
# dilation of images.
    import cv2
    import numpy as np


# Taking a matrix of size 5 as the kernel
    kernel = np.ones((5,5), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    return img
    
def autocontrast(img):
    import cv2
    import numpy as np

# read image


# normalize float versions
    norm_img1 = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_img2 = cv2.normalize(img, None, alpha=0, beta=1.2, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print(img)
# scale to uint8
    img = (255*norm_img1).astype(np.uint8)
    img = np.clip(norm_img2, 0, 1)
    img = (255*norm_img2).astype(np.uint8)

# write normalized output images

    return img
    
    
    

import cv2
import numpy as np
# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
	if (0 <= pix and pix <= r1):
		return (s1 / r1)*pix
	elif (r1 < pix and pix <= r2):
		return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
	else:
		return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

def contraststreching(img):
    
# Define parameters.
    r1 = 70
    s1 = 0
    r2 = 140
    s2 = 255

# Vectorize the function to apply it to each value in the Numpy array.
    pixelVal_vec = np.vectorize(pixelVal)

# Apply contrast stretching.
    img= pixelVal_vec(img, r1, s1, r2, s2)
    
    return img
def logarthemic(img):
    import cv2
    import numpy as np

# Open the image.

# Apply log transform.
    c = 255/(np.log(1 + np.max(img)))
    log_transformed = c * np.log(1 + img)

# Specify the data type.
    img= np.array(log_transformed, dtype = np.uint8)
    return img

import os
import numpy as np
import cv2

def exposure(img):
    #folder = 'source_folder'

# We get all the image files from the source folder
    #files = list([os.path.join(folder, f) for f in os.listdir(folder)])

# We compute the average by adding up the images
# Start from an explicitly set as floating point, in order to force the
# conversion of the 8-bit values from the images, which would otherwise overflow
    average = cv2.imread(img)
    
    img = cv2.imread(average)
    # NumPy adds two images element wise, so pixel by pixel / channel by channel
    average += img
 
# Divide by count (again each pixel/channel is divided)
    average /= len(img)

# Normalize the image, to spread the pixel intensities across 0..255
# This will brighten the image without losing information
    img = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)
        
    return img
    
import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
def curve(img):
    
    img = cv2.medianBlur(img,13)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,45,0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    dilate = cv2.dilate(thresh, kernel1, iterations=1)
    erode = cv2.erode(dilate, kernel,iterations=1)

# Remove small noise by filtering using contour area
    cnts = cv2.findContours(erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        if cv2.contourArea(c) < 800:
            if len(c)>0:
                cv2.drawContours(thresh,[c], 0, (0,0,0), -1)
        
# Compute Euclidean distance from every binary pixel
# to the nearest zero pixel then find peaks
    distance_map = ndimage.distance_transform_edt(erode)
    local_max = peak_local_max(distance_map, indices=False, min_distance=1, labels=thresh)

# Perform connected component analysis then apply Watershed
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=erode)

# Iterate through unique labels
    for label in np.unique(labels):
        if label == 0:
            continue

    # Create a mask
        mask = np.zeros(thresh.shape, dtype="uint8")
        mask[labels == label] = 255

    # Find contours and determine contour area
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        c = max(cnts, key=cv2.contourArea)
    
        cv2.drawContours(img, [c], -1, (36,255,12), -1)

        



    thresh = 155
    img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    return img

def greyscale(img):
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def heatfilter(img):
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2

    #img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(img) * 2**16).astype(np.uint16)[:,:,:3]
    img= cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    return img
    
    
import numpy as np
import cv2
from matplotlib import pyplot as plt
def grabcut(img):
    #img = cv2.imread('messi.jpg')
    #cv2.imshow('img', img)

    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (70,65,400,290)

    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    return img
    
def colourswap(img):
    import cv2 
    import numpy as np
    #img = cv2.imread('1.jpg') # Importing Sample Test Image
 
    lower_range = np.array([0,0,0])  # Set the Lower range value of color in BGR
    upper_range = np.array([100,70,255])   # Set the Upper range value of color in BGR
    mask = cv2.inRange(img,lower_range,upper_range) # Create a mask with range
    result = cv2.bitwise_and(img,img,mask = mask)  # Performing bitwise and operation with mask in img variable


    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Converting the Orginal image to Gray
    bw_bgr = cv2.cvtColor(bw,cv2.COLOR_GRAY2BGR) # Converting the Gray image to BGR format
    img= cv2.bitwise_or(bw_bgr,result) # Performing Bitwise OR operation with gray bgr image and previous result image
    return img
    
    
def warpfilter(img):
    import cv2
    import numpy as np
 

# Pixel values in original image
    red_point = [147,150]
    green_point = [256,182]
    black_point = [119,453]
    blue_point = [231,460]
 
# Create point matrix
    point_matrix = np.float32([red_point,green_point,black_point, blue_point])
 
# Draw circle for each point
    cv2.circle(img,(red_point[0],red_point[1]),10,(0,0,255),cv2.FILLED)
    cv2.circle(img,(green_point[0],green_point[1]),10,(0,255,0),cv2.FILLED)
    cv2.circle(img,(blue_point[0],blue_point[1]),10,(255,0,0),cv2.FILLED)
    cv2.circle(img,(black_point[0],black_point[1]),10,(0,0,0),cv2.FILLED)
 
# Output image size
    width, height = 250,350
 
# Desired points value in output images
    converted_red_pixel_value = [0,0]
    converted_green_pixel_value = [width,0]
    converted_black_pixel_value = [0,height]
    converted_blue_pixel_value = [width,height]
 
# Convert points
    converted_points = np.float32([converted_red_pixel_value,converted_green_pixel_value,
                               converted_black_pixel_value,converted_blue_pixel_value])
 
# perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix,converted_points)
    img = cv2.warpPerspective(img,perspective_transform,(width,height))
 
    return img
def backgroundsubstraction(img):
    import cv2
    
    sharp_img = cv2.createBackgroundSubtractorMOG2().apply(img)
    return img
    

def averagefilter(img):
    import cv2
    import numpy as np
  


    img = cv2.blur(img,(5,5))
    img = cv2.boxFilter(img, -1, (10, 10), normalize=True)
    
    return img  
    
def highpassfilter(img):
    import numpy as np
    import cv2


#edge detection filter
    kernel = np.array([[0.0, -1.0, 0.0], 
                   [-1.0, 4.0, -1.0],
                   [0.0, -1.0, 0.0]])

    kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

#filter the source image
    img = cv2.filter2D(img,-1,kernel)
    
    return img
    
def lowpassfilter(img):    
    import numpy as np
    import cv2

#read image
    

#prepare the 5x5 shaped filter
    kernel = np.array([[1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1], 
                   [1, 1, 1, 1, 1]])
    kernel = kernel/sum(kernel)

#filter the source image
    img= cv2.filter2D(img,-1,kernel)
    return img
def vignetefilter(img):   
    import numpy as np
    import cv2
	
	
#reading the image
    #input_image = cv2.imread('food.jpeg')
	
#resizing the image according to our need
# resize() function takes 2 parameters,
# the image and the dimensions
    img = cv2.resize(img, (480, 480))
	
# Extracting the height and width of an image
    rows, cols = img.shape[:2]
	
# generating vignette mask using Gaussian
# resultant_kernels
    X_resultant_kernel = cv2.getGaussianKernel(cols,200)
    Y_resultant_kernel = cv2.getGaussianKernel(rows,200)
	
#generating resultant_kernel matrix
    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
	
#creating mask and normalising by using np.linalg
# function
    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
    img = np.copy(img)
	

        
    return img
def fliprotationfilter(img):

    import cv2
 
# read image as grey scale
#img = cv2.imread('/home/arjun/Desktop/logos/python.png')
# get image height, width
    (h, w) = img.shape[:2]
# calculate the center of the image
    center = (w / 2, h / 2)
 
    angle90 = 90
    angle180 = 180
    angle270 = 270
 
    scale = 1.0
 
# Perform the counter clockwise rotation holding at the center
# 90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
 
# 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    img = cv2.warpAffine(img, M, (w, h))
 
# 270 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (h, w))
    #img= np.concatenate((img, rotated90, rotated180,rotated270), axis=1)
    return img

def translation(img):
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt

# read the input image
# convert from BGR to RGB so we can plot using matplotlib
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# disable x & y axis
    #plt.axis('off')
# show the image
    #plt.imshow(img)
    #plt.show()
# get the image shape
    rows, cols, dim = img.shape
# transformation matrix for translation
    M = np.float32([[1, 0, 50],
                [0, 1, 50],
                [0, 0, 1]])
# apply a perspective transformation to the image
    img = cv2.warpPerspective(img, M, (cols, rows))
    
    return img
    
    
def scaling(img):
 
    import numpy as np 
    import cv2
    img = cv2.resize(img, (320, 240))
     
    return img
def contourdetection(img):    
    import cv2 
    import numpy as np 

    #image = cv2.imread('bubblingFish.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    img= cv2.Canny(gray, 30, 200) 
    
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
   
    #print("Number of Contours: " + str(len(contours))) 
  
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    
    return img
    
    
def bestfilter(img):
    import cv2
    import numpy as np

    kernel = np.ones((5,5),np.uint8)
    print(kernel)



    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)
    imgCanny = cv2.Canny(imgBlur,100,200)
    imgDilation = cv2.dilate(imgCanny,kernel , iterations = 10)
    img = cv2.erode(imgDilation,kernel,iterations=2)
    
    return img

def blurdetection(img):
    # import the necessary packages
    from imutils import paths
# This is a customized library by https://www.pyimagesearch.com/
    import argparse
# The argparse module makes it easy to write user-friendly command-line interfaces. The program
# defines what arguments it requires, and argparse will figure out how to parse those out of sys.argv. 
# In this code, I don't use this, but you are welcome to implement it
    import cv2

# define threshold
    low_threshold = 350
    high_threshold = 1000

def variance_of_laplacian(img):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    #return cv2.Laplacian(img, cv2.CV_64F).var()

# load the image, convert it to grayscale, and compute the
# focus measure of the image using the Variance of Laplacian
# method
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Good"
 
# if the focus measure is less than the supplied threshold,
# then the image should be considered "blurry"
    if fm < low_threshold:
        text = "Blurry"

    elif fm > high_threshold:
        text = "Distorted"
    
    return img
def wienerfilter(img):
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import scipy.misc
    from scipy.misc import imread

    from skimage import color, restoration

    #image = imread('dog.jpg')
    astro = color.rgb2gray(img)

    from scipy.signal import convolve2d as conv2
    psf = np.ones((5, 5)) /25
    astro = conv2(astro, psf, 'same')
    astro += 0.1 * astro.std() * np.random.standard_normal(astro.shape)

    img, _ = restoration.unsupervised_wiener(astro, psf)
    
    return img
    
def instagramfilter(img):
    
    import cv2
    import sys
    import numpy as np

#read input image
#image=cv2.imread("../assets/anish.jpg")

    if img is None:
        print("can not find image")
        sys.exit()

#create a copy of input image to work on
    clarendon = img.copy()

#split the channels
    blueChannel, greenChannel, redChannel = cv2.split(clarendon)

#Interpolation values
    originalValues = np.array([0, 28, 56, 85, 113, 141, 170, 198, 227, 255])
    blueValues =     np.array([0, 38, 66, 104, 139, 175, 206, 226, 245, 255 ])
    redValues =      np.array([0, 16, 35, 64, 117, 163, 200, 222, 237, 249 ])
    greenValues =    np.array([0, 24, 49, 98, 141, 174, 201, 223, 239, 255 ])

#Creating the lookuptables
    fullRange = np.arange(0,256)
#Creating the lookuptable for blue channel
    blueLookupTable = np.interp(fullRange, originalValues, blueValues )
#Creating the lookuptables for green channel
    greenLookupTable = np.interp(fullRange, originalValues, greenValues )
#Creating the lookuptables for red channel
    redLookupTable = np.interp(fullRange, originalValues, redValues )

#Apply the mapping for blue channel
    blueChannel = cv2.LUT(blueChannel, blueLookupTable)
#Apply the mapping for green channel
    greenChannel = cv2.LUT(greenChannel, greenLookupTable)
#Apply the mapping for red channel
    redChannel = cv2.LUT(redChannel, redLookupTable)

#merging back the channels
    clarendon = cv2.merge([blueChannel, greenChannel, redChannel])

#convert to uint8
    img = np.uint8(clarendon)
    return img
    
    
    
import cv2
import math
import numpy as np

def dehaze(img):
    b,g,r = cv2.split(img)
    dc = cv2.min(cv2.min(r,g),b)
    sz=np.size(img)	
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    img = cv2.erode(dc,kernel)
  
    return img

#create windows to display images



