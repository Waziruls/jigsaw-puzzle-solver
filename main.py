import numpy as np
import cv2 as cv
import sys
import os
import glob
import piecesExtraction
import solving
import parametersSelect

def imgRescale(img, maxLen):
    h,w,_ = img.shape
    reductionFactor = 0.0
    if h > w:
        reductionFactor = h / maxLen
    else:
        reductionFactor = w / maxLen
    return cv.resize(img, (int(w/reductionFactor) ,int(h/reductionFactor)))

def showComparation(nombre,b1,b2,l1,l2):
    print(nombre + ":", solving.compararLados2(b1, b2, l1, l2))
    tempBack = np.zeros((600,200,3), np.uint8)
    cv.putText(tempBack,str(len(b1)) + " " + str(len(b2)),(30,25), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,cv.LINE_AA)
    cv.putText(tempBack,str(solving.compararLados2(b1, b2, l1, l2)),(40,75), cv.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),1,cv.LINE_AA)
    #print("---> compLados HSV:", solving.compararLadosRGB(b1, b2))
    b2 = b2[::-1]
    b1_d = []
    b2_d = []
    for i in b1:
        b1_d.extend([i[0], i[0]])
    for i in b2:
        b2_d.extend([i[0], i[0]])

    for i in range(0,20):
        if i < 10:
            tempBack[0:len(b1_d),i] = b1_d
            #tempBack[len(b1_d)-60][i] = (0,0,255)
        else:
            tempBack[0:len(b2_d),i] = b2_d
            #tempBack[len(b2_d)-60][i] = (0,0,255)
    cv.imshow(nombre, cv.cvtColor(tempBack, cv.COLOR_HSV2BGR))

def showAllBorders(piezas_lados, maxRes):
    maxLado = 0
    for i in range(0,len(piezas_lados)):
        for j in range(0,4):
            if len(piezas_lados[i][j]) > maxLado:
                maxLado = len(piezas_lados[i][j])
    tempBack = np.zeros((maxLado*2 + 20,len(piezas_lados)*4*10,3), np.uint8)
    cont = 0
    for i in range(0,len(piezas_lados)):
        for j in range(0,4):
            duplicated = []
            for b in piezas_lados[i][j]:
                duplicated.extend([b, b])
            for k in range(0,10):
                tempBack[0:len(duplicated),cont] = duplicated
                cont += 1
    tempBack = imgRescale(tempBack, maxRes)
    cv.imshow("All borders", cv.cvtColor(tempBack, cv.COLOR_HSV2BGR))

###############################################################################

img = None
puzzleH, puzzleW = 0, 0
chroma = None
maxRes = 0
ker_size = 0
alt_max_value = 0
mode = 0
tile = 100 # Size of pieces cut-offs

args = sys.argv[1:]
piezas_lados = []
piezas_shapes = []
piezas_imgs = []
piezas_lados_length = []
imgs_with_numbers = []
imgs = []

error = False
if len(args) >= 2:
    if args[0].isdigit() and args[1].isdigit() and int(args[0]) > 0 and int(args[1]) > 0:
        puzzleH, puzzleW = int(args[0]), int(args[1])
    else:
        error = True
else:
    error = True

if len(args) >= 3:
    if args[2].isdigit() and 0 <= int(args[2]) < 4:
        mode = int(args[2])
    else:
        error = True
else:
    mode = 0

if len(args) >= 4:
    if args[3].isdigit() and int(args[3]) > 0:
        jump_size = int(args[3])
    else:
        error = True
else:
    jump_size = 1

if len(args) >= 5:
    if args[4].isdigit() and int(args[4]) >= 0:
        alt_max_value = int(args[4])
    else:
        error = True
else:
    alt_max_value = 125

if len(args) >= 6:
    if args[5].isdigit() and int(args[5]) > 0:
        ker_size = int(args[5])
    else:
        error = True
else:
    ker_size = 7

if len(args) >= 7:
    if args[6].isdigit() and int(args[6]) >= 250:
        maxRes = int(args[6])
    else:
        error = True
else:
    maxRes = 1000

if os.path.isdir("./images"):
    files = glob.glob("./images/*.*")
    for file in files:
        parts = file.split(".")
        if parts[len(parts)-1] in ["jpg","png","jpeg","jpe","jp2","bmp","dib","pbm","pgm","ppm","sr","ras","tiff","tif","JPG","PNG","JPEG","JPE","BMP"]:
            imgs.append(cv.imread(file))
else:
    error = True

if not error and len(imgs) > 0:
    for img in imgs:
        img = imgRescale(img, 1280)
        chroma = parametersSelect.selectChroma(img, maxRes)
        components = piecesExtraction.getConnectedComponents(img, chroma)
        bordes, contours, cutImgs, imgWithNumbers = piecesExtraction.getBordes(components, img, chroma, ker_size) #bordes: [[b g r],[b g r],...]   #contours: [(y,x),(y,x),...]
        imgs_with_numbers.append(imgWithNumbers)
        p_lados, p_shapes, p_imgs, p_lados_length = parametersSelect.selectCorners(components, bordes, contours, cutImgs, tile, img)
        piezas_lados += p_lados
        piezas_shapes += p_shapes
        piezas_imgs += p_imgs
        piezas_lados_length += p_lados_length

    if len(piezas_lados) == puzzleH * puzzleW:
        print("Calculating borders differences...")
        difs_bordes, difs_bordes_fast = solving.calcularDifsBordes(piezas_lados, piezas_shapes, piezas_lados_length)
        print(str(len(difs_bordes)), "possible combinations")
        difs_bordes = np.array(difs_bordes)
        ind = np.argsort( difs_bordes[:,0] )
        difs_bordes = difs_bordes[ind]

        end = False
        tryagain = False
        while not end:
            solution = solving.resolverPuzzle(tile, piezas_imgs, piezas_lados, piezas_shapes, piezas_lados_length, difs_bordes, difs_bordes_fast, puzzleH, puzzleW, alt_max_value, maxRes, mode, jump_size)
            if mode > 0:
                cv.imshow("Solution positions", imgRescale(solution, maxRes))
                for i in range(0, len(imgs_with_numbers)):
                    cv.imshow("Pieces numbers (" + str(i) + ")", imgRescale(imgs_with_numbers[i], maxRes))
                showAllBorders(piezas_lados, maxRes)
                #showComparation("1", piezas_lados[0][0], piezas_lados[3][1], piezas_lados_length[0][0], piezas_lados_length[3][1])
                #showComparation("2", piezas_lados[9][1], piezas_lados[8][2], piezas_lados_length[9][1], piezas_lados_length[8][2])

            print("\nPress 'c' to try other parameters or press 'q' to quit.")
            while (True):
                if cv.waitKey() & 0xFF == ord('c'):
                    tryagain = True
                    break
                if cv.waitKey() & 0xFF == ord('q'):
                    end = True
                    break
            if tryagain:
                tryagain = False
                print("Introduce:")
                print("mode jump_size maxAlt maxRes (by default: 0 1 175 1000)")
                new_parameters = input()
                parts = new_parameters.split(" ")
                if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit() and parts[3].isdigit() and 0 <= int(parts[0]) < 4 and int(parts[1]) > 0 and int(parts[2]) >= 0 and int(parts[3]) >= 100:
                    mode, jump_size, alt_max_value, maxRes = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                elif len(parts) == 0:
                    mode, jump_size, alt_max_value, maxRes = 0, 1, 175, 1000
                else:
                    print("Error: Invalid parameters")
                    end = True
                    break
    else:
        cv.destroyAllWindows()
        print("Error:", str(len(piezas_lados)) + "pieces detected. Expected " + str(puzzleH * puzzleW))

    cv.destroyAllWindows()

else:
    cv.destroyAllWindows()
    print("Use instructions:")
    print("")
    print("  Input images must be in a folder named 'images' in the same directory as the program files.")
    print("  Press 'c' to continue afer adjusting chroma and corners.")
    print("")
    print("  Parameters:")
    print("    python main.py <Puzzle height> <Puzzle width> <Mode> <Jump size> <Max alternatives difference> <Inpaint kernel> <Max resolution> ")
    print("")
    print("    Puzzle height and Puzzle width:")
    print("      Heigth and width of the puzzle in number of pieces. Doesn't matter the order.")
    print("    Mode (OPTIONAL, default: 0):")
    print("      0 - Normal mode + reduced output")
    print("      1 - Normal mode + complete output")
    print("      2 - Normal mode + complete output + resolution animation")
    print("      3 - Step mode + complete output (Step mode will ask you if you want to continue searching for a better solution each time one is found).")
    print("          Step mode is recommended if the puzzle have more than 30 pieces and you don't want to increase jump size.")
    print("    Jump size (OPTIONAL, default: 1):")
    print("      If increased, it will try to start with less pieces. It's recommended to increase it if the program takes a long time.")
    print("      For example, if Jump size = 3, the algorithm will only try to start with pieces 0, 3, 6, 9,...")
    print("    Max alternatives difference (OPTIONAL, default: 125):")
    print("      Value that determines the maximum value difference to consider other piece an alternative to that position.")
    print("      If decreased, the algorithm will be more greedy. If you increse the value, the program will take longer but it will return better solutions.")
    print("      If you increase it too much, it will be difficult for the algorithm to place every piece.")
    print("      It is recommended to decrease it if the whole puzzle have a similar tonality.")
    print("    Inpaint kernel (OPTIONAL, default: 7):")
    print("      Size of the inpaint area. Increase it if noise appears in the extracted borders.")
    print("    Max resolution (OPTIONAL, default: 1000):")
    print("      Max resolution of the windows shown. Can't be less than 250.")
