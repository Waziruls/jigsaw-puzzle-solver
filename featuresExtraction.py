import numpy as np
import cv2 as cv
import math

PI = 3.1415926536

def distanceBetween(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def angle_between_points( p0, p1, p2 ):
  a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
  b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
  c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
  #r = ((a+b-c) / math.sqrt(4*a*b))
  #print("(a+b-c) / math.sqrt(4*a*b) =", r)
  return math.acos( (a+b-c) / math.sqrt(4*a*b) ) * 180/PI

def cornerVictor(borde, offset, angle):

    curvature = np.zeros(len(borde));
    for j in range(0,len(borde)):
        curvature[j] = angle_between_points(borde[j-offset], borde[j], borde[(j+offset)%len(borde)])

    corners = []
    cornersIndex = []
    for k in range(0,len(borde)):
        if (curvature[k] > angle):
            break

    #cv.circle(dst,(borde[k][1],borde[k][0]), 3, (255,255,255), -1)
    cont = 0
    for j in range(k,len(borde)+k+1):
        if (curvature[(j%len(borde))] <= angle):
            cont += 1
        elif cont > 0:
            index  = ((j%len(borde)) - ((cont//2)+1))%len(borde)
            corners.append((borde[index]))
            cornersIndex.append(index)
            #cv.circle(dst,(borde[index][1],borde[index][0]), 3, (255,0,255), -1)
            cont = 0

    return corners, cornersIndex, curvature

def drawCurvature(img, borde, curvature):
    for j in range(0,len(borde)):
        if (curvature[j] < 100):
            img[borde[j]] = (255,255,255)
        elif (curvature[j] < 120):
            img[borde[j]] = (0,0,255)
        elif (curvature[j] < 140):
            img[borde[j]] = (0,150,255)
        elif (curvature[j] < 160):
            img[borde[j]] = (0,255,255)
        elif (curvature[j] <= 180):
            img[borde[j]] = (0,255,0)
        else:
            img[borde[j]] = (255,0,0)

def drawCorners(img, corners):
    #cont = 0
    for c in corners:
        cv.circle(img,(c[1],c[0]), 3, (0,0,0), -1)
        #cv.putText(img,str(cont),(c[1],c[0]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv.LINE_AA)
        #cont += 1

def drawRectangle(img, rectangle):
    cv.line(img,tuple(reversed(rectangle[0])),tuple(reversed(rectangle[1])),(0,0,255),2)
    cv.line(img,tuple(reversed(rectangle[1])),tuple(reversed(rectangle[2])),(0,0,255),2)
    cv.line(img,tuple(reversed(rectangle[2])),tuple(reversed(rectangle[3])),(0,0,255),2)
    cv.line(img,tuple(reversed(rectangle[3])),tuple(reversed(rectangle[0])),(0,0,255),2)
    cv.line(img,tuple(reversed(rectangle[0])),tuple(reversed(rectangle[2])),(0,0,255),2)
    cv.line(img,tuple(reversed(rectangle[1])),tuple(reversed(rectangle[3])),(0,0,255),2)

def drawShape(img, contour, rectangleIndex):
    z = 0
    for ri in rectangleIndex:
        #cv.circle(img,(contour[ri][1],contour[ri][0]), 5, (255-(z*75),255-(z*75),255-(z*75)), -1)
        #cv.circle(img,(contour[ri][1],contour[ri][0]), 5, (255,255,255), -1)
        z += 1
    shapes = getFormaLado(contour, rectangleIndex)
    for i in range(0,4):
        y,x = 0,0
        if rectangleIndex[i] < rectangleIndex[(i+1)%4]:
            y,x = contour[int((rectangleIndex[i]+rectangleIndex[(i+1)%4])/2)]
        else:
            idx = int((((len(contour) + rectangleIndex[i])+rectangleIndex[(i+1)%4])/2)%len(contour))
            y,x = contour[idx]
        if shapes[i] == 2:
            cv.circle(img,(x,y), 3, (0,255,0), -1)
        elif shapes[i] == 1:
            cv.circle(img,(x,y), 3, (0,255,255), -1)
        else:
            cv.circle(img,(x,y), 3, (0,0,255), -1)

def distanceBetween(a, b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def lineEquationFrom2Points(p1, p2):
    x1 = p1[0]; y1 = p1[1]; x2 = p2[0]; y2 = p2[1]
    a, b, c = 0.0, 0.0, 0.0
    if (x2 == x1):
        a = 1.0
        b = 0.0
        c = -x1
    else:
        m = (y2-y1) / (x2-x1)
        a = m
        b = -1.0
        c = (-x1*m) + y1
    return a, b, c

def distanceBetweenLineAndPoint(p1, p2, p3):
    a, b, c = lineEquationFrom2Points(p1, p2)
    x0 = p3[0]; y0 = p3[1]
    d = abs(a*x0 + b*y0 + c) / math.sqrt(a**2 + b**2)
    return d

def distancesCompatible(cornersIndex, contour):
    distanceLimit = 18
    jumpSize = 4
    isCompatible = True
    for i in range(0,4):
        previous_d = 0
        surpassLimitCount = 0
        loseLimitCount = 0
        if (cornersIndex[i] < cornersIndex[((i+1)%4)]):
            for j in range(cornersIndex[i], cornersIndex[((i+1)%4)], jumpSize):
                d = distanceBetweenLineAndPoint(contour[cornersIndex[i]], contour[cornersIndex[((i+1)%4)]], contour[j])
                if previous_d < distanceLimit and d >= distanceLimit:
                    surpassLimitCount += 1
                if previous_d >= distanceLimit and d < distanceLimit:
                    loseLimitCount += 1
                previous_d = d
        else:
            for j in range(cornersIndex[i], len(contour), jumpSize):
                d = distanceBetweenLineAndPoint(contour[cornersIndex[i]], contour[cornersIndex[((i+1)%4)]], contour[j])
                if previous_d < distanceLimit and d >= distanceLimit:
                    surpassLimitCount += 1
                if previous_d >= distanceLimit and d < distanceLimit:
                    loseLimitCount += 1
                previous_d = d
            for j in range(0, cornersIndex[((i+1)%4)], jumpSize):
                d = distanceBetweenLineAndPoint(contour[cornersIndex[i]], contour[cornersIndex[((i+1)%4)]], contour[j])
                if previous_d < distanceLimit and d >= distanceLimit:
                    surpassLimitCount += 1
                if previous_d >= distanceLimit and d < distanceLimit:
                    loseLimitCount += 1
                previous_d = d
        if (surpassLimitCount != 1 and loseLimitCount != 1) and (surpassLimitCount != 0 and loseLimitCount != 0): # Si no es un lado con un solo saliente o entrante, o plano
            isCompatible = False
            break

    return isCompatible

def IsRectangle(a, b, c, d):
    x1=a[0]; y1=a[1]; x2=b[0]; y2=b[1]; x3=c[0]; y3=c[1]; x4=d[0]; y4=d[1];
    l21 = math.hypot(x2-x1, y2-y1)
    l43 = math.hypot(x4-x3, y4-y3)
    l32 = math.hypot(x3-x2, y3-y2)
    l14 = math.hypot(x1-x4, y1-y4)
    d31 = math.hypot(x3-x1, y3-y1)
    d42 = math.hypot(x4-x2, y4-y2)
    area = l21 * l32
    value = int((abs(l21 - l43)/max(l21, l43) + abs(l32 - l14)/max(l32, l14) + abs(d31 - d42)/max(d31, d42))*100)
    return area, value # 0 - 100, 0 is perfect rectangle


def biggerRectangleCorners(corners, cornersIndex, contour):

    c = 0
    bestArea = 0.0
    bestRectangle = None
    bestRectangleIndex = None
    for i in range(0, len(corners)-3):
        for j in range(i+1, len(corners)-2):
            for k in range(j+1, len(corners)-1):
                for l in range(k+1, len(corners)):
                    c += 1
                    area, rectValue = IsRectangle(corners[i], corners[j], corners[k], corners[l])
                    if rectValue < 18 and area > bestArea and distancesCompatible([cornersIndex[i], cornersIndex[j], cornersIndex[k], cornersIndex[l]], contour):
                        bestArea = area
                        bestRectangle = (corners[i], corners[j], corners[k], corners[l])
                        bestRectangleIndex = (cornersIndex[i],cornersIndex[j],cornersIndex[k],cornersIndex[l])

    return bestRectangle, bestRectangleIndex

def getLados(borde, rectangleIndex, contour):
    lados = []
    lados_length = []
    for i in range(0,4):
        lados_length.append(distanceBetween(contour[rectangleIndex[i]],contour[rectangleIndex[(i+1)%4]]))
        start = rectangleIndex[i]
        end = rectangleIndex[(i+1)%4]
        if start < end:
            lados.append(borde[start:end])
        else:
            lados.append(np.concatenate((borde[start:], borde[:end])))
    return lados, lados_length

def getFormaLado(contour, rectangleIndex):
    shapes = []
    x1 = contour[rectangleIndex[0]][1]
    x2 = contour[rectangleIndex[2]][1]
    y1 = contour[rectangleIndex[0]][0]
    y2 = contour[rectangleIndex[2]][0]
    xc,yc = int((x2 + x1)/2), int((y2+y1)/2) # Piece center

    for i in range(0,4):
        xl,yl = (contour[rectangleIndex[i]][1]+contour[rectangleIndex[(i+1)%4]][1]) / 2, (contour[rectangleIndex[i]][0]+contour[rectangleIndex[(i+1)%4]][0]) / 2 # Middle point between corners
        y,x = 0,0
        if rectangleIndex[i] < rectangleIndex[(i+1)%4]:
            y,x = contour[int((rectangleIndex[i]+rectangleIndex[(i+1)%4])/2)]
        else:
            idx = int((((len(contour) + rectangleIndex[i])+rectangleIndex[(i+1)%4])/2)%len(contour))
            y,x = contour[idx]
        midCornMidContDist = distanceBetween((xl,yl),(x,y))
        midPieceMidContDist = distanceBetween((xc,yc),(x,y))
        midPieceMidCornDist = distanceBetween((xc,yc),(xl,yl))
        if midCornMidContDist > 10 and midPieceMidContDist > midPieceMidCornDist:
            shapes.append(2) # Convexo
        elif midCornMidContDist > 10 and midPieceMidContDist < midPieceMidCornDist:
            shapes.append(0) # Concavo
        else:
            shapes.append(1) # Plano

    return shapes

def four_point_transform(image, pts):
	# obtain a consistent order of the points and add the vector
    # center-corner to leave a margin
    (tl, bl, br, tr) = pts
    x1,x2,y1,y2 = pts[0][1], pts[2][1], pts[0][0], pts[2][0]
    xc,yc = int((x2 + x1)/2), int((y2+y1)/2)

    vector_tl = (tl[1] - xc, tl[0] - yc)
    tl = (tl[1]+vector_tl[0], tl[0]+vector_tl[1])
    vector_tr = (tr[1] - xc, tr[0] - yc)
    tr = (tr[1]+vector_tr[0], tr[0]+vector_tr[1])
    vector_br = (br[1] - xc, br[0] - yc)
    br = (br[1]+vector_br[0], br[0]+vector_br[1])
    vector_bl = (bl[1] - xc, bl[0] - yc)
    bl = (bl[1]+vector_bl[0], bl[0]+vector_bl[1])

    rect = np.zeros((4, 2), dtype = "float32")
    rect[0][0], rect[0][1] = tl
    rect[1][0], rect[1][1] = tr
    rect[2][0], rect[2][1] = br
    rect[3][0], rect[3][1] = bl

	# compute the width of the new image
    widthA = distanceBetween(br,bl)
    widthB = distanceBetween(tr,tl)
    maxWidth = max(int(widthA), int(widthB))

	# compute the height of the new image
    heightA = distanceBetween(tr,br)
    heightB = distanceBetween(tl,bl)
    maxHeight = max(int(heightA), int(heightB))

	# construct the set of destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

	# compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

	# return the warped image
    return warped

def imgRescale(img, tile):
    h,w,_ = img.shape
    reductionFactor = 0.0
    if h > w:
        reductionFactor = h / tile
    else:
        reductionFactor = w / tile
    return cv.resize(img, (int(w/reductionFactor) ,int(h/reductionFactor)))

def getFeatures(components, bordes, contours, cutImgs, pixelOffset, angle, tile, img):

    imgCopy = np.copy(img)
    piezas_lados = [] # Where i'm going to save the 4 sides of each piece
    piezas_shapes = [] # 2-convex, 1-flat, 0-hollow
    piezas_imgs = []
    piezas_lados_length = []

    for i in range(0, len(components)):

        contour = contours[i]
        borde = bordes[i]

        corners,cornersIndex,curvature = cornerVictor(contour, pixelOffset, angle)

        drawCurvature(imgCopy, contour, curvature)
        drawCorners(imgCopy, corners)

        rectanglePoints, rectangleIndex = biggerRectangleCorners(corners, cornersIndex, contour)

        if rectanglePoints != None and len(rectanglePoints) == 4:
            drawRectangle(imgCopy, rectanglePoints)
            drawShape(imgCopy, contour, rectangleIndex)

            lados, lados_length = getLados(borde, rectangleIndex, contour)
            piezas_lados.append(lados)
            piezas_lados_length.append(lados_length)
            piezas_shapes.append(getFormaLado(contour, rectangleIndex))
            warped = four_point_transform(cutImgs[i], rectanglePoints)
            piezas_imgs.append(imgRescale(warped, tile))
            '''
            for ri in rectangleIndex:
                offset = 18
                cv.circle(imgCopy,(contour[ri-offset][1],contour[ri-offset][0]), 3, (255,255,255), -1)
                cv.circle(imgCopy,(contour[(ri+offset)%len(contour)][1],contour[(ri+offset)%len(contour)][0]), 3, (255,255,255), -1)
            '''
        i += 1

    return piezas_lados, piezas_shapes, piezas_imgs, piezas_lados_length, imgCopy
