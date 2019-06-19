import numpy as np
import cv2 as cv
from scipy import ndimage
import math

nImg = 0
nPieza = 0

def findLongestList(list):
    maxLen = 0
    maxLenIndex = 0
    if len(list) == 0:
        print("Error. Empty list.")
        return None
    if len(list) == 1:
        return 0, list[0]
    else:
        i = 0
        for l in list:
            if len(l) > maxLen:
                maxLen = len(l)
                maxLenIndex = i
            i += 1
        return maxLenIndex, (list[maxLenIndex])

def saveContourAsImage(path, contour, sourceImg):
    imgCon = np.zeros((contour.shape[0],1,3), np.uint8)
    for j in range(0,contour.shape[0]):
        imgCon[j][0] = sourceImg[contour[j][0][1]][contour[j][0][0]]
    cv.imwrite(path, imgCon)

def getConnectedComponents(img, chroma):
    ##### Mask the background color #####
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    print("Chroma:", chroma)
    # Threshold for an optimal value, it may vary depending on the image.
    mask = cv.inRange(hsv, chroma[0], chroma[1])
    mask = cv.bitwise_not(mask)
    #cv.imshow("mask", mask)

    ##### Binary opening and fill holes of the mask #####
    #gtf = ndimage.morphology.binary_opening(ndimage.binary_fill_holes(mask))
    gtf = mask

    ##### Separate connected components #####
    n,cc = cv.connectedComponents(gtf.astype(np.uint8).copy())
    #print("Components detected:", n)

    ##### Discard the smaller ones #####
    ok = [ cc==k for k in range(1,n) if np.sum(cc==k) > 1000]
    print("Pieces detected:", len(ok))
    return ok

def getBordes(ok, img, chroma, ker_size):
    global nImg, nPieza

    imgWithNumbers = np.copy(img)
    bordes = []
    goodContours = []
    cutImgs = []

    ##### Border extraction for each piece #####
    for i in range(0,len(ok)):
        ##### Create mask for the piece #####
        ciMask = ok[i].astype(np.uint8)  #convert to an unsigned byte
        ciMask*=255

        ##### Get countour of the mask/piece #####
        im2, contours, hierarchy = cv.findContours(ciMask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        ##### Create edge structure from contours #####
        edges = np.zeros((img.shape[0],img.shape[1]), np.uint8)
        for j in range(0,contours[0].shape[0]):
            edges[contours[0][j][0][1]][contours[0][j][0][0]] = (255)

        ##### Dilate edges and creating a mask from them #####
        ker = np.ones((ker_size, ker_size),np.uint8)
        maskC = cv.dilate(edges, ker)

        ##### Inpainting #####
        dst = cv.inpaint(img,maskC,4,cv.INPAINT_NS)
        #cv.imshow("dst" + str(i),dst)

        ##### Get only the piece in a black background #####
        cut = np.zeros_like(img, np.uint8)
        cut[ok[i]] = dst[ok[i]]
        #if i==12:
            #cv.imshow("dst" + str(i),cut)

        ##### Mask the background color again to eliminate a small border left #####
        hsv = cv.cvtColor(cut, cv.COLOR_BGR2HSV)

        maski = cv.inRange(hsv, chroma[0], chroma[1])
        maskBlack = cv.inRange(hsv, (0, 0, 0), (0, 0,0))
        maski = maski + maskBlack

        mask = cv.bitwise_not(maski)
        mask2 = np.copy(mask)

        ##### Get the piece again without that small background border #####
        maskbool = mask>0
        cut2 = np.zeros_like(img, np.uint8)
        cut2[maskbool] = dst[maskbool]

        ##### Countours, draw them black on the mask, and contours again #####
        im2, contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
        cv.drawContours(mask2, contours, -1, (0), 1)
        im2, contours2, hierarchy = cv.findContours(mask2,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)

        ##### Save cut image of the piece
        cv.drawContours(cut2, contours, -1, (0,0,0), 1)
        cutImgs.append(cut2)

        goodIndex, goodContour = findLongestList(contours2)
        #print(str(i), goodIndex)

        #cv.drawContours(imgWithNumbers, contours, -1, (255,255,255), 1)
        #cv.drawContours(imgWithNumbers, contours2, goodIndex, (255,0,0), 1)
        cv.circle(imgWithNumbers,(contours2[goodIndex][0][0][0],contours2[goodIndex][0][0][1]), 4, (0,255,0), -1)
        cv.putText(imgWithNumbers,str(nPieza),((contours2[goodIndex][0][0][0] + 10),(contours2[goodIndex][0][0][1] + 50)), cv.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2,cv.LINE_AA)
        nPieza += 1

        ##### Save the contour with the desire structure #####
        finalContour = []
        for j in range(0,goodContour.shape[0]):
            finalContour.append((goodContour[j][0][1],goodContour[j][0][0]))

        ##### Save the border's rgb values of each piece #####
        array = np.zeros((len(finalContour),1,3), np.uint8)
        for j in range(0,len(finalContour)):
            array[j][0] = dst[finalContour[j][0]][finalContour[j][1]]
        array = cv.cvtColor(array, cv.COLOR_BGR2HSV)
        bordes.append(array);
        goodContours.append(finalContour)

        #saveContourAsImage("./bordes/contour_" + str(i) + ".png", goodContour, dst)

    nImg += 1
    return bordes, goodContours, cutImgs, imgWithNumbers
