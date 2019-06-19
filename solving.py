import numpy as np
import cv2 as cv

def imgRescale(img, maxLen):
    h,w,_ = img.shape
    reductionFactor = 0.0
    if h > w:
        reductionFactor = h / maxLen
    else:
        reductionFactor = w / maxLen
    return cv.resize(img, (int(w/reductionFactor) ,int(h/reductionFactor)))

def compararLados2(b1, b2, b1_length, b2_length):
    flatSector = 15
    ratio = 0
    if len(b1) < len(b2):
        temp = b1
        b1 = b2
        b2 = temp
    b2 = b2[::-1]

    dif_bordes = 0
    for i in range(0, flatSector):
        h_dif = min(abs(int(b1[i][0][0]) - int(b2[i][0][0])), (abs(int(min(b1[i][0][0], b2[i][0][0]) + 255-max(b1[i][0][0], b2[i][0][0])))))
        sv_dif = abs(int(b1[i][0][1]) - int(b2[i][0][1])) + abs(int(b1[i][0][2]) - int(b2[i][0][2]))
        dif_bordes = dif_bordes + h_dif + sv_dif

    for i in range(len(b2) - flatSector, len(b2)):
        h_dif = min(abs(int(b1[i][0][0]) - int(b2[i][0][0])), (abs(int(min(b1[i][0][0], b2[i][0][0]) + 255-max(b1[i][0][0], b2[i][0][0])))))
        sv_dif = abs(int(b1[i][0][1]) - int(b2[i][0][1])) + abs(int(b1[i][0][2]) - int(b2[i][0][2]))
        dif_bordes = dif_bordes + h_dif + sv_dif

    ratio = (len(b1) + 2*flatSector) / (len(b2) + 2*flatSector)
    for i in range(flatSector, len(b2) - flatSector):
        b1_i = int(i * ratio)
        h_dif = min(abs(int(b1[b1_i][0][0]) - int(b2[i][0][0])), (abs(int(min(b1[b1_i][0][0], b2[i][0][0]) + 255-max(b1[b1_i][0][0], b2[i][0][0])))))
        sv_dif = abs(int(b1[b1_i][0][1]) - int(b2[i][0][1])) + abs(int(b1[b1_i][0][2]) - int(b2[i][0][2]))
        dif_bordes = dif_bordes + h_dif + sv_dif

    lengthDif = max(b1_length, b2_length) / min(b1_length, b2_length)
    return int((dif_bordes/len(b2))*10*lengthDif)

def areCompatible(shapes1, shapes2, s1, s2): # 2-convex, 1-flat, 0-hollow
    shape1 = shapes1[s1]
    shape2 = shapes2[s2]
    shape1_next = shapes1[((s1+1)%4)]
    shape2_next = shapes2[((s2+1)%4)]
    shape1_prev = shapes1[((s1-1)%4)]
    shape2_prev = shapes2[((s2-1)%4)]
    if shape1==1 or shape2==1:
        return False
    if shape1==shape2:
        return False
    if (shape1_next == 1 and shape2_prev != 1) or (shape2_next == 1 and shape1_prev != 1) or (shape1_prev == 1 and shape2_next != 1) or (shape2_prev == 1 and shape1_next != 1):
        return False
    return True

def calcularDifsBordes(piezas_lados, piezas_shapes, piezas_lados_length):
    difs_bordes = []
    npiezas = len(piezas_lados)
    difs_bordes_fast = np.zeros((npiezas, npiezas, 4, 4))
    difs_bordes_fast.fill(9999)
    for i in range(0,npiezas-1): # Todos con todos
        for k in range (i+1,npiezas):
            for j in range(0,4):
                for l in range(0,4):
                    if areCompatible(piezas_shapes[i], piezas_shapes[k], j, l):
                        value = compararLados2(piezas_lados[i][j], piezas_lados[k][l], piezas_lados_length[i][j], piezas_lados_length[k][l])
                        difs_bordes_fast[i][k][j][l] = int(value)
                        difs_bordes_fast[k][i][l][j] = int(value)
                        difs_bordes.append((value, i, k, j, l))
    return difs_bordes, difs_bordes_fast

def cacularPosPieza2(pieza1, pieza2, borde1, borde2, x, y, rotacionPiezas, piezas_imgs):

    # Por si la pieza colocada fue rotada, hallar donde esta ahora el borde que buscamos
    borde1 = borde1 + rotacionPiezas[pieza1]
    borde1 = borde1 % 4

    rotacionPieza2 = 0

    #print ("borde1:",borde1," borde2:",borde2)

    if (borde1) == 1 and (borde2) == 3:
        imgToPrint = piezas_imgs[pieza2]
        posPieza2y = y + 1; posPieza2x = x
    elif (borde1) == 3 and (borde2) == 1:
        imgToPrint = piezas_imgs[pieza2]
        posPieza2y = y - 1; posPieza2x = x
    elif (borde1) == 2 and (borde2) == 0:
        imgToPrint = piezas_imgs[pieza2]
        posPieza2y = y; posPieza2x = x + 1
    elif (borde1) == 0 and (borde2) == 2:
        imgToPrint = piezas_imgs[pieza2]
        posPieza2y = y; posPieza2x = x - 1

    elif (borde1) == 3 and (borde2) == 3:
        imgToPrint = np.rot90(piezas_imgs[pieza2],2)
        rotacionPieza2 = 2
        posPieza2y = y - 1; posPieza2x = x
    elif (borde1) == 2 and (borde2) == 2:
        imgToPrint = np.rot90(piezas_imgs[pieza2],2)
        rotacionPieza2 = 2
        posPieza2y = y; posPieza2x = x + 1
    elif (borde1) == 1 and (borde2) == 1:
        imgToPrint = np.rot90(piezas_imgs[pieza2],2)
        rotacionPieza2 = 2
        posPieza2y = y + 1; posPieza2x = x
    elif (borde1) == 0 and (borde2) == 0:
        imgToPrint = np.rot90(piezas_imgs[pieza2],2)
        rotacionPieza2 = 2
        posPieza2y = y; posPieza2x = x - 1

    elif (borde1) == 3 and (borde2) == 2:
        imgToPrint = np.rot90(piezas_imgs[pieza2],3)
        rotacionPieza2 = 3
        posPieza2y = y - 1; posPieza2x = x
    elif (borde1) == 2 and (borde2) == 3:
        imgToPrint = np.rot90(piezas_imgs[pieza2],1)
        rotacionPieza2 = 1
        posPieza2y = y; posPieza2x = x + 1
    elif (borde1) == 2 and (borde2) == 1:
        imgToPrint = np.rot90(piezas_imgs[pieza2],3)
        rotacionPieza2 = 3
        posPieza2y = y; posPieza2x = x + 1
    elif (borde1) == 1 and (borde2) == 2:
        imgToPrint = np.rot90(piezas_imgs[pieza2],1)
        rotacionPieza2 = 1
        posPieza2y = y + 1; posPieza2x = x
    elif (borde1) == 1 and (borde2) == 0:
        imgToPrint = np.rot90(piezas_imgs[pieza2],3)
        rotacionPieza2 = 3
        posPieza2y = y + 1; posPieza2x = x
    elif (borde1) == 0 and (borde2) == 1:
        imgToPrint = np.rot90(piezas_imgs[pieza2],1)
        rotacionPieza2 = 1
        posPieza2y = y; posPieza2x = x - 1
    elif (borde1) == 0 and (borde2) == 3:
        imgToPrint = np.rot90(piezas_imgs[pieza2],3)
        rotacionPieza2 = 3
        posPieza2y = y; posPieza2x = x - 1
    elif (borde1) == 3 and (borde2) == 0:
        imgToPrint = np.rot90(piezas_imgs[pieza2],1)
        rotacionPieza2 = 1
        posPieza2y = y - 1; posPieza2x = x

    return imgToPrint, posPieza2y, posPieza2x, rotacionPieza2

def whereIs(pieza, tablero):
    a = zip(*np.where(tablero == pieza))
    b = ((list(a))[0])
    return b # Coordenadas de donde esta pieza1 en el tablero

def dimensionesPuzzle(tablero):
    h,w = tablero.shape
    ha = np.zeros(h)
    wa = np.zeros(w)
    minI = -1
    minJ = -1
    for i in range(0,h):
        for j in range (0,w):
            if tablero[i][j] != -1:
                ha[i] = 1; wa[j] = 1
                if minI == -1:
                    minI = i
                elif i < minI:
                    minI = i
                if minJ == -1:
                    minJ = j
                elif j < minJ:
                    minJ = j

    return np.count_nonzero(ha), np.count_nonzero(wa), (minI, minJ)

def dimensionesMinimasSolucion(tablero, piezas_shapes, rotacionPiezas):
    h,w = tablero.shape
    minI, maxI, minJ, maxJ = -1, -1, -1, -1
    for i in range(0,h):
        for j in range (0,w):
            if tablero[i][j] != -1:
                if minI == -1 and piezas_shapes[tablero[i][j]][3 - rotacionPiezas[tablero[i][j]]] == 1:
                    minI = i
                elif minI == -1:
                    minI = i-1
            if tablero[h-i-1][j] != -1:
                if maxI == -1 and piezas_shapes[tablero[h-i-1][j]][1 - rotacionPiezas[tablero[h-i-1][j]]] == 1:
                    maxI = h-i-1
                elif maxI == -1:
                    maxI = h-i

    for j in range(0,w):
        for i in range (0,h):
            if tablero[i][j] != -1:
                if minJ == -1 and piezas_shapes[tablero[i][j]][0 - rotacionPiezas[tablero[i][j]]] == 1:
                    minJ = j
                elif minJ == -1:
                    minJ = j-1
            if tablero[i][w-j-1] != -1:
                if maxJ == -1 and piezas_shapes[tablero[i][w-j-1]][2 - rotacionPiezas[tablero[i][w-j-1]]] == 1:
                    maxJ = w-j-1
                elif maxJ == -1:
                    maxJ = w-j

    #print("mini maxi minj maxj",minI, maxI, minJ, maxJ)
    return maxI-minI+1, maxJ-minJ+1

def isShapeCompatible(tablero, puzzleLimits, rotacionPiezas, piezas_shapes, piezas_lados, piezas_lados_length, maxDifBordes, pieza2, posPieza2x, posPieza2y, rotacionPieza2, difs_bordes_fast):

    # if the piece has an edge and there is another piece in the same row/column with a no-edge oriented the same way, return False
    if piezas_shapes[pieza2][0 - rotacionPieza2] == 1:
        for i in range(0,tablero.shape[0]):
            if tablero[i][posPieza2x] != -1 and tablero[i][posPieza2x] != pieza2 and piezas_shapes[(tablero[i][posPieza2x])][0 - rotacionPiezas[tablero[i][posPieza2x]]] != 1:
                return False
    if piezas_shapes[pieza2][1 - rotacionPieza2] == 1:
        for i in range(0,tablero.shape[1]):
            if tablero[posPieza2y][i] != -1 and tablero[posPieza2y][i] != pieza2 and piezas_shapes[(tablero[posPieza2y][i])][1 - rotacionPiezas[tablero[posPieza2y][i]]] != 1:
                return False
    if piezas_shapes[pieza2][2 - rotacionPieza2] == 1:
        for i in range(0,tablero.shape[0]):
            if tablero[i][posPieza2x] != -1 and tablero[i][posPieza2x] != pieza2 and piezas_shapes[(tablero[i][posPieza2x])][2 - rotacionPiezas[tablero[i][posPieza2x]]] != 1:
                return False
    if piezas_shapes[pieza2][3 - rotacionPieza2] == 1:
        for i in range(0,tablero.shape[1]):
            if tablero[posPieza2y][i] != -1 and tablero[posPieza2y][i] != pieza2 and piezas_shapes[(tablero[posPieza2y][i])][3 - rotacionPiezas[tablero[posPieza2y][i]]] != 1:
                return False

    # if the piece have a neighbour, check if the shapes are compatible, and also they are not a very bad union
    upPiece = tablero[posPieza2y-1][posPieza2x]
    if upPiece != -1:
        upNeighbourDownShape = piezas_shapes[int(upPiece)][int((1 - rotacionPiezas[upPiece])%4)]
        myUpShape = piezas_shapes[pieza2][int((3 - rotacionPieza2)%4)]
        difBordes = difs_bordes_fast[upPiece][pieza2][int((1 - rotacionPiezas[upPiece])%4)][int((3 - rotacionPieza2)%4)]
        if myUpShape == 1 or upNeighbourDownShape == 1 or (myUpShape == upNeighbourDownShape) or difBordes > maxDifBordes:
            return False

    rightPiece = tablero[posPieza2y][posPieza2x+1]
    if rightPiece != -1:
        rightNeighbourLeftShape = piezas_shapes[int(rightPiece)][int((0 - rotacionPiezas[rightPiece])%4)]
        myRightShape = piezas_shapes[pieza2][int((2 - rotacionPieza2)%4)]
        difBordes = difs_bordes_fast[rightPiece][pieza2][int((0 - rotacionPiezas[rightPiece])%4)][int((2 - rotacionPieza2)%4)]
        if myRightShape == 1 or rightNeighbourLeftShape == 1 or (myRightShape == rightNeighbourLeftShape) or difBordes > maxDifBordes:
            return False
    leftPiece = tablero[posPieza2y][posPieza2x-1]
    if leftPiece != -1:
        leftNeighbourRightShape = piezas_shapes[int(leftPiece)][int((2 - rotacionPiezas[leftPiece])%4)]
        myLeftShape = piezas_shapes[pieza2][int((0 - rotacionPieza2)%4)]
        difBordes = difs_bordes_fast[leftPiece][pieza2][int((2 - rotacionPiezas[leftPiece])%4)][int((0 - rotacionPieza2)%4)]
        if myLeftShape == 1 or leftNeighbourRightShape == 1 or (myLeftShape == leftNeighbourRightShape) or difBordes > maxDifBordes:
            return False

    downPiece = tablero[posPieza2y+1][posPieza2x]
    if downPiece != -1:
        downNeighbourUpShape = piezas_shapes[int(downPiece)][int((3 - rotacionPiezas[downPiece])%4)]
        myDownShape = piezas_shapes[pieza2][int((1 - rotacionPieza2)%4)]
        difBordes = difs_bordes_fast[downPiece][pieza2][int((3 - rotacionPiezas[downPiece])%4)][int((1 - rotacionPieza2)%4)]
        if myDownShape == 1 or downNeighbourUpShape == 1 or (myDownShape == downNeighbourUpShape) or difBordes > maxDifBordes:
            return False

    return True

def isLimitCompatible(puzzleLimits, piezas_shapes, puzzleH, puzzleW, pieza2, rotacionPieza2, posPieza2x, posPieza2y, hMin, wMin):

    # if the piece has a plain side and its puzzle limit has not been set, but the opposite is, check that the distance isnt less than the minor puzzle axis
    if puzzleLimits[2] == -1 and piezas_shapes[pieza2][0 - rotacionPieza2] == 1 and puzzleLimits[3] != -1:
        if puzzleLimits[3] - posPieza2x + 1 < min(puzzleH, puzzleW):
            return False
    if puzzleLimits[1] == -1 and piezas_shapes[pieza2][1 - rotacionPieza2] == 1 and puzzleLimits[0] != -1:
        if posPieza2y - puzzleLimits[0] + 1 < min(puzzleH, puzzleW):
            return False
    if puzzleLimits[3] == -1 and piezas_shapes[pieza2][2 - rotacionPieza2] == 1 and puzzleLimits[2] != -1:
        if posPieza2x - puzzleLimits[2] + 1 < min(puzzleH, puzzleW):
            return False
    if puzzleLimits[0] == -1 and piezas_shapes[pieza2][3 - rotacionPieza2] == 1 and puzzleLimits[1] != -1:
        if puzzleLimits[1] - posPieza2y + 1 < min(puzzleH, puzzleW):
            return False

    # if the piece is gonna be inserted in a puzzle edge row/column, check if it has a plain side oriented accordingly
    if posPieza2x == puzzleLimits[2] and piezas_shapes[pieza2][0 - rotacionPieza2] != 1:
        return False
    if posPieza2y == puzzleLimits[1] and piezas_shapes[pieza2][1 - rotacionPieza2] != 1:
        return False
    if posPieza2x == puzzleLimits[3] and piezas_shapes[pieza2][2 - rotacionPieza2] != 1:
        return False
    if posPieza2y == puzzleLimits[0] and piezas_shapes[pieza2][3 - rotacionPieza2] != 1:
        return False

    # if the piece has an edge and the puzzle limit associated to its orientation is established, check if its in that limit row/column
    if piezas_shapes[pieza2][0 - rotacionPieza2] == 1 and puzzleLimits[2] != -1 and posPieza2x != puzzleLimits[2]:
        return False
    if piezas_shapes[pieza2][1 - rotacionPieza2] == 1 and puzzleLimits[1] != -1 and posPieza2y != puzzleLimits[1]:
        return False
    if piezas_shapes[pieza2][2 - rotacionPieza2] == 1 and puzzleLimits[3] != -1 and posPieza2x != puzzleLimits[3]:
        return False
    if piezas_shapes[pieza2][3 - rotacionPieza2] == 1 and puzzleLimits[0] != -1 and posPieza2y != puzzleLimits[0]:
        return False

    if hMin > max(puzzleH, puzzleW) or wMin > max(puzzleH, puzzleW):
        return False

    # si el eje y esta establecido y es el eje grande, mirar que no se pase la anchura
    if puzzleLimits[0] != -1 and puzzleLimits[1] != -1:
        if puzzleLimits[1] - puzzleLimits[0] + 1 == max(puzzleH, puzzleW):
            if wMin > min(puzzleH, puzzleW):
                return False

    # si el eje x esta establecido y es el eje grande, mirar que no se pase la altura
    if puzzleLimits[2] != -1 and puzzleLimits[3] != -1:
        if puzzleLimits[3] - puzzleLimits[2] + 1 == max(puzzleH, puzzleW):
            if hMin > min(puzzleH, puzzleW):
                return False

    return True

def updatePuzzleLimits(tablero, puzzleLimits, puzzleH, puzzleW, hMin, wMin, piezas_shapes, pieza2, rotacionPieza2, posPieza2x, posPieza2y):
    # if you dont know the inferior/superior limit for the height/width and the inserted piece have a border that discovers it, set that limit
    if puzzleLimits[2] == -1 and piezas_shapes[pieza2][0 - rotacionPieza2] == 1:
        puzzleLimits[2] = posPieza2x
    if puzzleLimits[1] == -1 and piezas_shapes[pieza2][1 - rotacionPieza2] == 1:
        puzzleLimits[1] = posPieza2y
    if puzzleLimits[3] == -1 and piezas_shapes[pieza2][2 - rotacionPieza2] == 1:
        puzzleLimits[3] = posPieza2x
    if puzzleLimits[0] == -1 and piezas_shapes[pieza2][3 - rotacionPieza2] == 1:
        puzzleLimits[0] = posPieza2y

    # if the minimun height for the solution is greater than the smallest axis, then the height is the high axis and the width the small one
    if hMin > min(puzzleH, puzzleW):
        if puzzleLimits[1] != -1:
            puzzleLimits[0] = puzzleLimits[1] - (max(puzzleH, puzzleW)-1)
        elif puzzleLimits[0] != -1:
            puzzleLimits[1] = puzzleLimits[0] + (max(puzzleH, puzzleW)-1)
        if puzzleLimits[3] != -1:
            puzzleLimits[2] = puzzleLimits[3] - (min(puzzleH, puzzleW)-1)
        elif puzzleLimits[2] != -1:
            puzzleLimits[3] = puzzleLimits[2] + (min(puzzleH, puzzleW)-1)
    # if the minimun width for the solution is greater than the smallest axis, then the width is the high axis and the height the small one
    if wMin > min(puzzleH, puzzleW):
        if puzzleLimits[3] != -1:
            puzzleLimits[2] = puzzleLimits[3] - (max(puzzleH, puzzleW)-1)
        elif puzzleLimits[2] != -1:
            puzzleLimits[3] = puzzleLimits[2] + (max(puzzleH, puzzleW)-1)
        if puzzleLimits[1] != -1:
            puzzleLimits[0] = puzzleLimits[1] - (min(puzzleH, puzzleW)-1)
        elif puzzleLimits[0] != -1:
            puzzleLimits[1] = puzzleLimits[0] + (min(puzzleH, puzzleW)-1)

    # if you know both limits for the height, then you check if its the high or the small axis and set the others limits accordingly
    if puzzleLimits[0] != -1 and puzzleLimits[1] != -1:
        if puzzleLimits[1] - puzzleLimits[0] + 1 == min(puzzleH, puzzleW):
            if puzzleLimits[3] != -1:
                puzzleLimits[2] = puzzleLimits[3] - (max(puzzleH, puzzleW)-1)
            elif puzzleLimits[2] != -1:
                puzzleLimits[3] = puzzleLimits[2] + (max(puzzleH, puzzleW)-1)
            elif wMin == max(puzzleH, puzzleW): # if you dont know any of the width limits but min width is already the max axis, set both width limits
                _,_,pos = dimensionesPuzzle(tablero)
                puzzleLimits[2] = pos[1] - 1
                puzzleLimits[3] = puzzleLimits[2] + max(puzzleH, puzzleW) - 1
        else:
            if puzzleLimits[3] != -1:
                puzzleLimits[2] = puzzleLimits[3] - (min(puzzleH, puzzleW)-1)
            elif puzzleLimits[2] != -1:
                puzzleLimits[3] = puzzleLimits[2] + (min(puzzleH, puzzleW)-1)
            elif wMin == min(puzzleH, puzzleW): # if you dont know any of the width limits but min width is already the min axis, set both width limits
                _,_,pos = dimensionesPuzzle(tablero)
                puzzleLimits[2] = pos[1] - 1
                puzzleLimits[3] = puzzleLimits[2] + min(puzzleH, puzzleW) - 1

    # if you know both limits for the width, then you check if its the high or the small axis and set the others limits accordingly
    if puzzleLimits[2] != -1 and puzzleLimits[3] != -1:
        if puzzleLimits[3] - puzzleLimits[2] + 1 == min(puzzleH, puzzleW):
            if puzzleLimits[1] != -1:
                puzzleLimits[0] = puzzleLimits[1] - (max(puzzleH, puzzleW)-1)
            elif puzzleLimits[0] != -1:
                puzzleLimits[1] = puzzleLimits[0] + (max(puzzleH, puzzleW)-1)
            elif hMin == max(puzzleH, puzzleW): # if you dont know any of the height limits but min height is already the max axis, set both height limits
                _,_,pos = dimensionesPuzzle(tablero)
                puzzleLimits[0] = pos[0] - 1
                puzzleLimits[1] = puzzleLimits[0] + max(puzzleH, puzzleW) - 1
        else:
            if puzzleLimits[1] != -1:
                puzzleLimits[0] = puzzleLimits[1] - (min(puzzleH, puzzleW)-1)
            elif puzzleLimits[0] != -1:
                puzzleLimits[1] = puzzleLimits[0] + (min(puzzleH, puzzleW)-1)
            elif hMin == min(puzzleH, puzzleW): # if you dont know any of the height limits but min height is already the min axis, set both height limits
                _,_,pos = dimensionesPuzzle(tablero)
                puzzleLimits[0] = pos[0] - 1
                puzzleLimits[1] = puzzleLimits[0] + min(puzzleH, puzzleW) - 1

    # si las dimensiones minimas superan en cualquier eje a la dimension pequeña del puzzle (ya sabes que ese eje es el grande),
    # si ademas es ya igual a la dimension grande y no hay ningun limite puesto aun, poner los dos limites a la vez
    # si el otro eje es ya dimesion minima la dimension pequeña del puzzle, y no hay ningun limite puesto aun, poner los dos a la vez

def hasEdge(pieza2, piezas_shapes):
    for i in range(0,4):
        if piezas_shapes[pieza2][i] == 1:
            return True
    return False

def areAlternatives(maxDif, relacionesUsadas, piezasColocadas, difs_bordes, pieza1, borde1, j):
    alternatives = []
    jalternatives = []
    j_ = 0
    while j_ in range(0,j):
        # Si la relacion no esta usada y una de las piezas de la misma es la que estamos tratando y la otra no esta ya en piezasColocadas:
        if (relacionesUsadas[j_] == 0 and (((difs_bordes[j_][1] == pieza1 and difs_bordes[j_][3] == borde1) and (not (difs_bordes[j_][2] in piezasColocadas)))
                                        or ((difs_bordes[j_][2] == pieza1 and difs_bordes[j_][4] == borde1) and (not (difs_bordes[j_][1] in piezasColocadas))))):
            alternatives.append(difs_bordes[j_])
            jalternatives.append(j_)
        j_ += 1

    j_ = j+1
    while j_ < len(relacionesUsadas) and difs_bordes[j_][0] < difs_bordes[j][0] + maxDif:
        if (relacionesUsadas[j_] == 0 and (((difs_bordes[j_][1] == pieza1 and difs_bordes[j_][3] == borde1) and (not (difs_bordes[j_][2] in piezasColocadas)))
                                        or ((difs_bordes[j_][2] == pieza1 and difs_bordes[j_][4] == borde1) and (not (difs_bordes[j_][1] in piezasColocadas))))):
            alternatives.append(difs_bordes[j_])
            jalternatives.append(j_)
        j_ += 1
    #print(str(len(piezasColocadas)+1)+". Alternativas para", pieza1, borde1, ":", len(alternatives), alternatives)

    moreAlternatives = False
    if len(alternatives) > 0:
        moreAlternatives = True

    return moreAlternatives, alternatives, jalternatives

# Returns the number of neighbors of posPieza2, their piece number, and wich of them is pieza1
def neighbors(posPieza2x, posPieza2y, tablero, pieza1):
    n = 0
    ni = -1
    neighbors = np.zeros(4, np.int8) # left, down, right, up
    neighbors.fill(-1)
    if posPieza2y-1 >= 0 and tablero[posPieza2y-1][posPieza2x] != -1:
        n += 1
        neighbors[3] = tablero[posPieza2y-1][posPieza2x]
        if tablero[posPieza2y-1][posPieza2x] == pieza1:
            ni = 3
    if posPieza2y+1 < tablero.shape[0] and tablero[posPieza2y+1][posPieza2x] != -1:
        n += 1
        neighbors[1] = tablero[posPieza2y+1][posPieza2x]
        if tablero[posPieza2y+1][posPieza2x] == pieza1:
            ni = 1
    if posPieza2x-1 >= 0 and tablero[posPieza2y][posPieza2x-1] != -1:
        n += 1
        neighbors[0] = tablero[posPieza2y][posPieza2x-1]
        if tablero[posPieza2y][posPieza2x-1] == pieza1:
            ni = 0
    if posPieza2x+1 < tablero.shape[1] and tablero[posPieza2y][posPieza2x+1] != -1:
        n += 1
        neighbors[2] = tablero[posPieza2y][posPieza2x+1]
        if tablero[posPieza2y][posPieza2x+1] == pieza1:
            ni = 2
    return n, neighbors, ni

def evaluateSolution(tablero, rotacionPiezas, difs_bordes_fast):
    piecesValues = []
    h, w, p = dimensionesPuzzle(tablero)
    for i in range(p[0], p[0]+h):
        for j in range(p[1], p[1]+w):
            if tablero[i][j] != -1:
                sumaTotalPieza = 0
                upPiece = tablero[i-1][j]
                if upPiece != -1:
                    sumaTotalPieza += difs_bordes_fast[upPiece][tablero[i][j]][int((1 - rotacionPiezas[upPiece])%4)][int((3 - rotacionPiezas[tablero[i][j]])%4)]
                rightPiece = tablero[i][j+1]
                if rightPiece != -1:
                    sumaTotalPieza += difs_bordes_fast[rightPiece][tablero[i][j]][int((0 - rotacionPiezas[rightPiece])%4)][int((2 - rotacionPiezas[tablero[i][j]])%4)]
                leftPiece = tablero[i][j-1]
                if leftPiece != -1:
                    sumaTotalPieza += difs_bordes_fast[leftPiece][tablero[i][j]][int((2 - rotacionPiezas[leftPiece])%4)][int((0 - rotacionPiezas[tablero[i][j]])%4)]
                downPiece = tablero[i+1][j]
                if downPiece != -1:
                    sumaTotalPieza += difs_bordes_fast[downPiece][tablero[i][j]][int((3 - rotacionPiezas[downPiece])%4)][int((1 - rotacionPiezas[tablero[i][j]])%4)]

                if sumaTotalPieza > 0:
                    piecesValues.append(sumaTotalPieza)

    return (sum(piecesValues) / len(rotacionPiezas))

def makeCompactedSolution(solutionTablero, piezasRotations, piezas_imgs, maxRes):
    h, w, pos = dimensionesPuzzle(solutionTablero)
    tablero = np.zeros((h,w), np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            tablero[i][j] = solutionTablero[pos[0] + i][pos[1] + j]

    imgs200px = []
    for i in range(0, len(piezas_imgs)):
        imgs200px.append(cv.resize(piezas_imgs[i], (200,200)))
        #cv.imwrite("./test/"+str(i)+".png", imgs200px[i])

    back = np.zeros((h*100+100,w*100+100,3), np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            if tablero[i][j] != -1 and tablero[i][j] < len(imgs200px):
                img = imgs200px[tablero[i][j]]
                if piezasRotations[tablero[i][j]] == 1:
                    img = np.rot90(img,1)
                elif piezasRotations[tablero[i][j]] == 2:
                    img = np.rot90(img,2)
                elif piezasRotations[tablero[i][j]] == 3:
                    img = np.rot90(img,3)
                mask = cv.inRange(img, (0,0,0), (0,0,0))
                mask = cv.bitwise_not(mask)
                for k in range(0, img.shape[0]):
                    for l in range(0, img.shape[1]):
                        if mask[k][l] > 0:
                            back[k+i*100][l+j*100] = img[k][l]
    if maxRes and maxRes > 0:
        back = imgRescale(back, maxRes)
    return back

def puzzleAnimation(windowName, solutionTablero, piezasRotations, piezas_imgs, piezasColocadas, maxRes):

    h, w, pos = dimensionesPuzzle(solutionTablero)
    tablero = np.zeros((h,w), np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            tablero[i][j] = solutionTablero[pos[0] + i][pos[1] + j]

    imgs200px = []
    for i in range(0, len(piezas_imgs)):
        imgs200px.append(cv.resize(piezas_imgs[i], (200,200)))
        #cv.imwrite("./test/"+str(i)+".png", imgs200px[i])

    back = np.zeros((h*100+100,w*100+100,3), np.uint8)

    nPieza = 0
    counter = 0
    while(True):
        if cv.waitKey() & 0xFF == ord('q') or cv.waitKey() & 0xFF == ord('c') or nPieza == len(piezasColocadas):
            break
        if counter == 10:
            index = np.where(tablero == piezasColocadas[nPieza])
            i = index[0][0]
            j = index[1][0]
            if tablero[i][j] < len(imgs200px):
                img = imgs200px[tablero[i][j]]
                if piezasRotations[tablero[i][j]] == 1:
                    img = np.rot90(img,1)
                elif piezasRotations[tablero[i][j]] == 2:
                    img = np.rot90(img,2)
                elif piezasRotations[tablero[i][j]] == 3:
                    img = np.rot90(img,3)
                mask = cv.inRange(img, (0,0,0), (0,0,0))
                mask = cv.bitwise_not(mask)
                for k in range(0, img.shape[0]):
                    for l in range(0, img.shape[1]):
                        if mask[k][l] > 0:
                            back[k+i*100][l+j*100] = img[k][l]
            nPieza += 1
            counter = 0
        if maxRes and maxRes > 0:
            back_resized = imgRescale(back, maxRes)
        cv.imshow(windowName, back_resized)
        counter += 1
    cv.destroyWindow(windowName)

def algoritmo(back, tile, npiezas, piezas_imgs, piezas_lados, piezas_shapes, piezas_lados_length, tablero, puzzleLimits, piezasColocadas, rotacionPiezas, relacionesUsadas, difs_bordes, difs_bordes_fast, puzzleH, puzzleW, alt_max_value):
    while npiezas > len(piezasColocadas):
        j = 0
        while j < len(relacionesUsadas):
            #print("////////////////////////// Inicio while /////////////////////////////////")
            j = 0
            colocada = False
            while not colocada:
                while j < len(relacionesUsadas) and relacionesUsadas[j] == 1:
                    j = j + 1
                if j==len(relacionesUsadas):
                    #print("FIN DE difs_bordes ALCANZADO")
                    break
                colocada = False
                pieza1 = int(difs_bordes[j][1])
                pieza2 = int(difs_bordes[j][2])
                for k in range(0, len(piezasColocadas)):
                    # Si no estan ya las dos colocadas y es solo una de las dos
                    if (pieza1 == piezasColocadas[k] or pieza2 == piezasColocadas[k]) and not (pieza1 in piezasColocadas and pieza2 in piezasColocadas):
                        borde1 = difs_bordes[j][3]
                        borde2 = difs_bordes[j][4]
                        # Para que la pieza1 sea siempre la que está ya en el tablero y la pieza2 la que quieres poner
                        if pieza2 == piezasColocadas[k]:
                            temp = pieza1
                            pieza1 = pieza2
                            pieza2 = temp
                            borde2 = difs_bordes[j][3]
                            borde1 = difs_bordes[j][4]

                        yPieza1 ,xPieza1 = whereIs(pieza1, tablero)
                        imgToPrint, posPieza2y, posPieza2x, rotacionPieza2 = cacularPosPieza2(pieza1, pieza2, borde1, borde2, xPieza1, yPieza1, rotacionPiezas, piezas_imgs)
                        y_offset = posPieza2y * tile; x_offset = posPieza2x * tile

                         # Si es una posición dontro del tablero y vacía
                        if posPieza2y < tablero.shape[0] and posPieza2y >= 0 and posPieza2x < tablero.shape[1] and posPieza2x >= 0 and tablero[posPieza2y, posPieza2x] == -1:
                            tablero[posPieza2y, posPieza2x] = pieza2
                            rotacionPiezas[pieza2] = rotacionPieza2
                            yt, xt, _ = dimensionesPuzzle(tablero)
                            maxDifBordes = max(difs_bordes[int(difs_bordes.shape[0]*0.60)][0], difs_bordes[j][0])

                            hMin, wMin = dimensionesMinimasSolucion(tablero, piezas_shapes, rotacionPiezas)
                            isLimitComp = isLimitCompatible(puzzleLimits, piezas_shapes, puzzleH, puzzleW, pieza2, rotacionPieza2, posPieza2x, posPieza2y, hMin, wMin)
                            isSHapeComp = isShapeCompatible(tablero, puzzleLimits, rotacionPiezas, piezas_shapes, piezas_lados, piezas_lados_length, maxDifBordes, pieza2, posPieza2x, posPieza2y, rotacionPieza2, difs_bordes_fast)
                            if isSHapeComp and isLimitComp:
                                ##################### ALTERNATIVAS ########################################################
                                areAlt, alternatives, jalternatives = areAlternatives(alt_max_value, relacionesUsadas, piezasColocadas, difs_bordes, pieza1, borde1, j)
                                if areAlt:
                                    alternatives.append(difs_bordes[j]) # Adding pieza2 to alternatives
                                    jalternatives.append(j)
                                    totalNbs, nbs, ni = neighbors(posPieza2x, posPieza2y, tablero, pieza1)
                                    if totalNbs > 1:
                                        mejorSumaAlt = 0
                                        mejorAlt = -1
                                        altCounter = 0
                                        for alt in alternatives:
                                            sumaAlt = 0
                                            rot = 0
                                            if ni == 0:
                                                if alt[2] == nbs[0]:
                                                    _, _, _, rot = cacularPosPieza2(nbs[0], alt[1], (2 - rotacionPiezas[nbs[0]]) % 4, alt[3], posPieza2x - 1, posPieza2y, rotacionPiezas, piezas_imgs)
                                                else:
                                                    _, _, _, rot = cacularPosPieza2(nbs[0], alt[2], (2 - rotacionPiezas[nbs[0]]) % 4, alt[4], posPieza2x - 1, posPieza2y, rotacionPiezas, piezas_imgs)
                                            elif ni == 1:
                                                if alt[2] == nbs[1]:
                                                    _, _, _, rot = cacularPosPieza2(nbs[1], alt[1], (3 - rotacionPiezas[nbs[1]]) % 4, alt[3], posPieza2x, posPieza2y + 1, rotacionPiezas, piezas_imgs)
                                                else:
                                                    _, _, _, rot = cacularPosPieza2(nbs[1], alt[2], (3 - rotacionPiezas[nbs[1]]) % 4, alt[4], posPieza2x, posPieza2y + 1, rotacionPiezas, piezas_imgs)
                                            elif ni == 2:
                                                if alt[2] == nbs[2]:
                                                    _, _, _, rot = cacularPosPieza2(nbs[2], alt[1], (0 - rotacionPiezas[nbs[2]]) % 4, alt[3], posPieza2x + 1, posPieza2y, rotacionPiezas, piezas_imgs)
                                                else:
                                                    _, _, _, rot = cacularPosPieza2(nbs[2], alt[2], (0 - rotacionPiezas[nbs[2]]) % 4, alt[4], posPieza2x + 1, posPieza2y, rotacionPiezas, piezas_imgs)
                                            elif ni == 3:
                                                if alt[2] == nbs[3]:
                                                    _, _, _, rot = cacularPosPieza2(nbs[3], alt[1], (1 - rotacionPiezas[nbs[3]]) % 4, alt[3], posPieza2x, posPieza2y - 1, rotacionPiezas, piezas_imgs)
                                                else:
                                                    _, _, _, rot = cacularPosPieza2(nbs[3], alt[2], (1 - rotacionPiezas[nbs[3]]) % 4, alt[4], posPieza2x, posPieza2y - 1, rotacionPiezas, piezas_imgs)

                                            if nbs[0] != -1: # Left neighbour
                                                leftNb = nbs[0]
                                                leftNbLado = (2 - rotacionPiezas[leftNb]) % 4
                                                if alt[1] == pieza1:
                                                    sumaAlt += difs_bordes_fast[alt[2]][leftNb][(0 - rot) % 4][leftNbLado]
                                                else:
                                                    sumaAlt += difs_bordes_fast[alt[1]][leftNb][(0 - rot) % 4][leftNbLado]

                                            if nbs[1] != -1: # Down neighbour
                                                downNb = nbs[1]
                                                downNbLado = (3 - rotacionPiezas[downNb]) % 4
                                                if alt[1] == pieza1:
                                                    sumaAlt += difs_bordes_fast[alt[2]][downNb][(1 - rot) % 4][downNbLado]
                                                else:
                                                    sumaAlt += difs_bordes_fast[alt[1]][downNb][(1 - rot) % 4][downNbLado]

                                            if nbs[2] != -1: # Right neighbour
                                                rightNb = nbs[2]
                                                rightNbLado = (0 - rotacionPiezas[rightNb]) % 4
                                                if alt[1] == pieza1:
                                                    sumaAlt += difs_bordes_fast[alt[2]][rightNb][(2 - rot) % 4][rightNbLado]
                                                else:
                                                    sumaAlt += difs_bordes_fast[alt[1]][rightNb][(2 - rot) % 4][rightNbLado]
                                            if nbs[3] != -1: # Up neighbour
                                                upNb = nbs[3]
                                                upNbLado = (1 - rotacionPiezas[upNb]) % 4
                                                if alt[1] == pieza1:
                                                    sumaAlt += difs_bordes_fast[alt[2]][upNb][(3 - rot) % 4][upNbLado]
                                                else:
                                                    sumaAlt += difs_bordes_fast[alt[1]][upNb][(3 - rot) % 4][upNbLado]

                                            if mejorSumaAlt == 0 or sumaAlt < mejorSumaAlt:
                                                mejorSumaAlt = sumaAlt
                                                mejorAlt = jalternatives[altCounter]
                                            altCounter += 1

                                        # marcar TODAS las demas menos la elegida y poner la j en la elegida
                                        for altIndex in jalternatives:
                                            if altIndex != mejorAlt:
                                                relacionesUsadas[altIndex] = 1
                                        j = mejorAlt + 1

                                    tablero[posPieza2y, posPieza2x] = -1
                                    break
                                ##################### ALTERNATIVAS ########################################################
                                piezasColocadas.append(pieza2)
                                rotacionPiezas[pieza2] = rotacionPieza2
                                updatePuzzleLimits(tablero, puzzleLimits, puzzleH, puzzleW, hMin, wMin, piezas_shapes, pieza2, rotacionPieza2, posPieza2x, posPieza2y)
                                back[y_offset:y_offset+imgToPrint.shape[0], x_offset:x_offset+imgToPrint.shape[1]] = imgToPrint
                                cv.putText(back,str(pieza2),(x_offset+10,y_offset+20), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv.LINE_AA)
                                cv.putText(back,str(len(piezasColocadas)),(x_offset+80,y_offset+20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),1,cv.LINE_AA)
                                cv.putText(back,str((0-rotacionPieza2)%4),(x_offset+15,y_offset+50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv.LINE_AA)
                                cv.putText(back,str((1-rotacionPieza2)%4),(x_offset+45,y_offset+80), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv.LINE_AA)
                                cv.putText(back,str((2-rotacionPieza2)%4),(x_offset+75,y_offset+50), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv.LINE_AA)
                                cv.putText(back,str((3-rotacionPieza2)%4),(x_offset+45,y_offset+20), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv.LINE_AA)
                                relacionesUsadas[j] = 1
                                colocada = True
                                j = 0
                            else:
                                tablero[posPieza2y, posPieza2x] = -1
                                # No es una pieza compatible
                                relacionesUsadas[j] = 1
                            break
                        else:
                            # Marcar linea (Sitio ocupado o fuera del tablero)
                            relacionesUsadas[j] = 1

                    # Fin del if de si alguna de las dos piezas esta colocada
                # Fin del for de las piezasColocadas
                j = j+1
            # Fin del while not colocada
        # Fin del for de las filas de dif_Bordes
        break
    # Fin del while not piezasColocadas = nPiezas
    hSol, wSol, pSol = dimensionesPuzzle(tablero)
    solution = back[pSol[0]*tile:pSol[0]*tile+hSol*tile, pSol[1]*tile:pSol[1]*tile+wSol*tile]
    return solution, tablero, piezasColocadas, rotacionPiezas

def resolverPuzzle(tile, piezas_imgs, piezas_lados, piezas_shapes, piezas_lados_length, difs_bordes, difs_bordes_fast, puzzleH, puzzleW, alt_max_value, maxRes, mode, jump_size):

    solutions = []
    bestSolution = 0
    bestSolutionValue = -1.0
    bestSolutionTablero = []
    bestSolutionRotations = []
    bestPiezasColocadas = []
    solCount = 0
    newBestSol = False

    nPieza = 0
    end = False
    while not end:

        print("Trying with piece", nPieza)
        # Variables de apoyo para la resolucion del puzzle
        maxPuzzleAxis = max(puzzleH, puzzleW)
        back = np.zeros(((maxPuzzleAxis*2+1)*100,(maxPuzzleAxis*2+1)*100,3), np.uint8)
        bh,bw,_ = back.shape
        bhn,bwn = bh//tile, bw//tile
        tablero = np.zeros((bhn, bwn), np.int8)
        tablero.fill(-1)
        piezasColocadas = []
        rotacionPiezas = np.zeros(len(piezas_imgs), np.uint8)
        relacionesUsadas = np.zeros(difs_bordes.shape[0], np.uint8)
        puzzleLimits = np.zeros(4, np.int8) # 0 - min h, 1 - max h, 2 - min w, 3 - max w
        puzzleLimits.fill(-1)

        # Coloco la primera pieza en el centro del fondo
        x_offset = (bwn//2 * tile)
        y_offset = (bhn//2 * tile)
        pieza1 = nPieza
        priPieza = piezas_imgs[pieza1]
        rotacionPiezas[pieza1] = 0
        back[y_offset:y_offset+priPieza.shape[0], x_offset:x_offset+priPieza.shape[1]] = priPieza
        cv.putText(back,str(pieza1),(x_offset+10,y_offset+20), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2,cv.LINE_AA)
        tablero[bhn//2, bwn//2] = pieza1
        piezasColocadas.append(pieza1)
        hMin, wMin = dimensionesMinimasSolucion(tablero, piezas_shapes, rotacionPiezas)
        updatePuzzleLimits(tablero, puzzleLimits, puzzleH, puzzleW, hMin, wMin, piezas_shapes, pieza1, 0, bwn//2, bhn//2)
        solution, tablero, piezasColocadas, rotacionPiezas = algoritmo(back, tile, len(piezas_imgs), piezas_imgs, piezas_lados, piezas_shapes, piezas_lados_length, tablero, puzzleLimits, piezasColocadas, rotacionPiezas, relacionesUsadas, difs_bordes, difs_bordes_fast, puzzleH, puzzleW, alt_max_value)
        solutions.append(solution)
        #cv.imshow("Solution positions " + str(nPieza), solution)
        #cv.imshow("Solution " + str(nPieza), makeCompactedSolution(tablero, rotacionPiezas, piezas_imgs, maxRes))
        print(len(piezasColocadas), "pieces placed")
        if len(piezasColocadas) == len(piezas_imgs):
            solutionValue = evaluateSolution(tablero, rotacionPiezas, difs_bordes_fast)
            print("Possible solution found with value", solutionValue)
            if solutionValue < bestSolutionValue or bestSolutionValue < 0:
                print("-- New best solution --")
                bestSolutionValue = solutionValue
                bestSolution = solCount
                bestSolutionTablero = tablero
                bestSolutionRotations = rotacionPiezas
                bestPiezasColocadas = piezasColocadas
                newBestSol = True
        elif len(piezasColocadas) > len(bestPiezasColocadas):
            bestSolution = solCount
            bestSolutionTablero = tablero
            bestSolutionRotations = rotacionPiezas
            bestPiezasColocadas = piezasColocadas

        solCount += 1

        if newBestSol and mode == 3:
            newBestSol = False
            compSol = makeCompactedSolution(bestSolutionTablero, bestSolutionRotations, piezas_imgs, maxRes)
            print("\nPress 'c' to continue or press 's' to stop.")
            while (True):
                cv.imshow("Best solution", compSol)
                if cv.waitKey() & 0xFF == ord('c'):
                    break
                if cv.waitKey() & 0xFF == ord('s'):
                    end = True
                    break

        nPieza += jump_size
        if nPieza >= len(piezas_imgs):
            end = True

    if len(bestPiezasColocadas) > 0 and mode == 2:
        puzzleAnimation("Puzzle animation", bestSolutionTablero, bestSolutionRotations, piezas_imgs, bestPiezasColocadas, maxRes)

    if len(bestPiezasColocadas) > 0:
        compSol = makeCompactedSolution(bestSolutionTablero, bestSolutionRotations, piezas_imgs, maxRes)
        cv.imshow("Best Solution", compSol)

    return solutions[bestSolution]
