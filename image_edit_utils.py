import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import util
from skimage import feature
from skimage import color
from skimage import filters

from scipy import ndimage
from sklearn import cluster


def rm_white_frame (im):

    # create a negative (to invert the white frame into black, to fill with 0)
    im = 1-im

    h,w,d = im.shape
    #left limit
    for i in range(w):
        if np.sum(im[:,i,:]) > 0:
            break
    #right limit
    for j in range(w-1,0,-1):
        if np.sum(im[:,j,:]) > 0:
            break

    #top limit
    for k in range(h):
        if np.sum(im[k,:,:]) > 0:
            break
    #bottom limit
    for l in range(h-1,0,-1):
        if np.sum(im[l,:,:]) > 0:
            break
    
    cropped = im[k:l+1,i:j+1,:].copy() 
    #back to normal
    cropped = 1- cropped

    return (cropped)    

def segmentation(gray_image): 

    image = color.gray2rgb(gray_image)

    x, y, z = image.shape
    image_2d = image.reshape(x*y, z)
    image_2d.shape

    kmeans_cluster = cluster.KMeans(n_clusters=4)
    kmeans_cluster.fit(image_2d)
    cluster_centers = kmeans_cluster.cluster_centers_
    cluster_labels = kmeans_cluster.labels_
    image_segmented = cluster_centers[cluster_labels].reshape(x, y, z)

    return (image_segmented)

    plt.show()



def rm_bg2(gray_img): 


    _, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]


    img_contours = sorted(img_contours, key=cv2.contourArea)

   
    for i in img_contours:
        if cv2.contourArea(i) > 10000000000:
            break


    mask = np.zeros(gray_img.shape[:2], np.uint8)
    cv2.drawContours(mask, [i],-1, 255, -1)

    # mask[np.where(mask==255)]=100
    # mask[np.where(mask==0)]=255
    # mask[np.where(mask==100)]=0

    new_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    return (mask)


def filter_edges(edges, gray_image):
    print (np.max(gray_image))
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if (i-1>0 and i<edges.shape[0]-1):
                if (gray_image[i-1,j]<130 and gray_image[i+1,j]<130):
                    edges[i,j]=False
        
    return (edges)

