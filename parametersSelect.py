import cv2 as cv
import numpy as np
import featuresExtraction

maxRes = 1080

def imgRescale(img, maxLen):
    h, w = img.shape[0], img.shape[1]
    reductionFactor = 0.0
    if h > w:
        reductionFactor = h / maxLen
    else:
        reductionFactor = w / maxLen
    return cv.resize(img, (int(w/reductionFactor) ,int(h/reductionFactor)))

title_window = 'Chroma Adjustment'
chromaSelected = False
vals = [0,0,0]
h,s,v = 0,0,0
y_dif_ratio = 1
x_dif_ratio = 1

def on_mouse(event,x,y,flags,param):
    global chromaSelected, h,s,v, y_dif_ratio, x_dif_ratio
    if not chromaSelected and event == cv.EVENT_LBUTTONDOWN:
        h,s,v = param[int(y*y_dif_ratio)][int(x*x_dif_ratio)]
        print("Chroma base color selected (hsv):", h,s,v)
        chromaSelected = True

def on_trackbarH(val):
    global hsv, vals
    vals[0] = val

def on_trackbarS(val):
    global hsv, vals
    vals[1] = val

def on_trackbarV(val):
    global hsv, vals
    vals[2] = val

def selectChroma(img, maxResolution):
    global maxRes, chromaSelected, y_dif_ratio, x_dif_ratio
    maxRes = maxResolution
    img_res = imgRescale(img, maxRes)
    y_dif_ratio = img.shape[0] / img_res.shape[0]
    x_dif_ratio = img.shape[1] / img_res.shape[1]
    chromaSelected = False
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    cv.namedWindow(title_window)

    while(not chromaSelected):
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.setMouseCallback(title_window, on_mouse, hsv)
        cv.imshow(title_window, imgRescale(img, maxRes))

    trackbar_nameH = 'h  %d +/-' % h
    trackbar_nameS = 's  %d +/-' % s
    trackbar_nameV = 'v  %d +/-' % v
    cv.createTrackbar(trackbar_nameH, title_window , 0, 50, on_trackbarH)
    cv.createTrackbar(trackbar_nameS, title_window , 0, 150, on_trackbarS)
    cv.createTrackbar(trackbar_nameV, title_window , 0, 150, on_trackbarV)
    on_trackbarH(0)
    on_trackbarS(0)
    on_trackbarV(0)

    c1,c2 = (0,0,0),(0,0,0)
    mask = None
    while(True):
        if cv.waitKey(1) & 0xFF == ord('c'):
            break
        c1 = h-vals[0],s-vals[1],v-vals[2]
        c1 = tuple(map(int, c1))
        c2 = h+vals[0],s+vals[1],v+vals[2]
        c2 = tuple(map(int, c2))
        mask = cv.inRange(hsv, c1, c2)
        mask = cv.bitwise_not(mask)
        cv.imshow(title_window, imgRescale(mask, maxRes))

    cv.destroyWindow(title_window)
    print(c1, c2)
    return (c1, c2)

################################################################################

pixelOffset, angle = 14, 125
piezas_lados = [] # Where i'm going to save the 4 sides of each piece
piezas_shapes = [] # 2-convex, 1-flat, 0-hollow
piezas_imgs = []
piezas_lados_length = []
imgCopy = None

components = None
bordes = None
contours = None
cutImgs = None
tile = None
img = None

title_window_corners = "Corner Parameters Adjustment"

def on_trackbar_angle(val):
    global piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length, pixelOffset, angle, imgCopy
    angle = val + 100
    piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length, imgCopy = featuresExtraction.getFeatures(components, bordes, contours, cutImgs, pixelOffset, angle, tile, img)
    cv.imshow(title_window_corners, imgRescale(imgCopy, maxRes))

def on_trackbar_offset(val):
    global piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length, pixelOffset, angle, imgCopy
    pixelOffset = val + 5
    piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length, imgCopy = featuresExtraction.getFeatures(components, bordes, contours, cutImgs, pixelOffset, angle, tile, img)
    cv.imshow(title_window_corners, imgRescale(imgCopy, maxRes))

def selectCorners(components_, bordes_, contours_, cutImgs_, tile_, img_):
    global imgCopy, components, bordes, contours, cutImgs, tile, img
    components, bordes, contours, cutImgs, tile, img = components_, bordes_, contours_, cutImgs_, tile_, img_
    imgCopy = np.copy(img)
    cv.namedWindow(title_window_corners)
    trackbar_name_angle = 'angle +100'
    cv.createTrackbar(trackbar_name_angle, title_window_corners , 25, 50, on_trackbar_angle)
    trackbar_name_offset = 'offset +5'
    cv.createTrackbar(trackbar_name_offset, title_window_corners , 10, 45, on_trackbar_offset)
    on_trackbar_offset(10)
    while(True):
        if cv.waitKey(1) & 0xFF == ord('c'):
            cv.destroyWindow(title_window_corners)
            break
    #cv.imshow("Features", imgCopy)
    return piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length
