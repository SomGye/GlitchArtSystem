import cv2
import numpy as np
import random
import math

#Author: Maxwell Crawford
#CSC475 Final: Glitch Art Generator
#10-29-17 10:44pm

'''
Use:
assignment1 -> sobelx, sobely parts -> NEED TO MODIFY for color, randomness?
assignment2 -> harris corner BONUS part
assignment5 -> anaglyph (with other color combos, shifted offsets vertically)

'''
## Define Functions
# Define Sobel X Filters:
def sobelX(case, x, y, sobelXmatrix, grayimage):
    '''
    :param case: an int, 1 thru 5, of Sobel cases
    :param x: matrix iteration x-location
    :param y: matrix iteration y-location
    :return result: the resulting value to put into new Sobel X matrix (temp)
    '''
    sumFxM = 0 #Sum of (Filter value * Matrix value)
    sumFilters = 0 #Sum of Filter Values
    result = 0
    if (case == 1):
        sumFilters = sobelXmatrix[0][0] + sobelXmatrix[0][1] + \
                     sobelXmatrix[1][0] + sobelXmatrix[1][1]
        # Iterate a 2x2 grid:
        for i in range(2):
            for j in range(2):
                sumFxM += int((sobelXmatrix[i][j] * grayimage[x + i][y + j]))
    elif (case == 2):
        sumFilters = sobelXmatrix[0][1] + sobelXmatrix[0][2] + \
                     sobelXmatrix[1][1] + sobelXmatrix[1][2]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelXmatrix[0][2] * grayimage[x][y])) #center
        sumFxM += int((sobelXmatrix[0][1] * grayimage[x][y - 1])) #left1
        sumFxM += int((sobelXmatrix[1][2] * grayimage[x + 1][y])) #down1
        sumFxM += int((sobelXmatrix[1][1] * grayimage[x + 1][y-1]))  # down1, left1
    elif (case == 3):
        sumFilters = sobelXmatrix[1][0] + sobelXmatrix[1][1] + \
                     sobelXmatrix[2][0] + sobelXmatrix[2][1]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelXmatrix[2][0] * grayimage[x][y]))  # center
        sumFxM += int((sobelXmatrix[2][1] * grayimage[x][y + 1]))  # right1
        sumFxM += int((sobelXmatrix[1][0] * grayimage[x - 1][y]))  # up1
        sumFxM += int((sobelXmatrix[1][1] * grayimage[x - 1][y + 1]))  # up1, right1
    elif (case == 4):
        sumFilters = sobelXmatrix[1][1] + sobelXmatrix[1][2] + \
                     sobelXmatrix[2][1] + sobelXmatrix[2][2]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelXmatrix[2][2] * grayimage[x][y]))  # center
        sumFxM += int((sobelXmatrix[2][1] * grayimage[x][y - 1]))  # left1
        sumFxM += int((sobelXmatrix[1][2] * grayimage[x - 1][y]))  # up1
        sumFxM += int((sobelXmatrix[1][1] * grayimage[x - 1][y - 1]))  # up1, left1
    elif (case == 5):
        '''
        NOTE: Total of Sobel Filter values cancels to 0, so we need a base value.
        '''
        sumFilters = 1
        # Iterate full 3x3 grid manually:
        sumFxM += int((sobelXmatrix[0][0] * grayimage[x - 1][y - 1])) #up1, left1
        sumFxM += int((sobelXmatrix[0][1] * grayimage[x][y - 1])) #up1
        sumFxM += int((sobelXmatrix[0][2] * grayimage[x + 1][y - 1])) #up1, right1
        sumFxM += int((sobelXmatrix[1][0] * grayimage[x - 1][y])) #left1
        sumFxM += int((sobelXmatrix[1][1] * grayimage[x][y])) #center
        sumFxM += int((sobelXmatrix[1][2] * grayimage[x + 1][y])) #right1
        sumFxM += int((sobelXmatrix[2][0] * grayimage[x - 1][y + 1])) #down1, left1
        sumFxM += int((sobelXmatrix[2][1] * grayimage[x][y + 1])) #down1
        sumFxM += int((sobelXmatrix[2][2] * grayimage[x + 1][y + 1])) #down1, right1

    result = int(sumFxM / sumFilters)
    if (result < 0): #flip sign of negative
        result *= -1
    return result

# Define Sobel Y Filters:
def sobelY(case, x, y, sobelYmatrix, grayimage):
    '''
    :param case: an int, 1 thru 5, of Sobel cases
    :param x: matrix iteration x-location
    :param y: matrix iteration y-location
    :return result: the resulting value to put into new Sobel Y matrix (temp)
    '''
    sumFxM = 0 #Sum of (Filter value * Matrix value)
    sumFilters = 0 #Sum of Filter Values
    result = 0
    if (case == 1):
        sumFilters = sobelYmatrix[0][0] + sobelYmatrix[0][1] + \
                     sobelYmatrix[1][0] + sobelYmatrix[1][1]
        # Iterate a 2x2 grid:
        for i in range(2):
            for j in range(2):
                sumFxM += int((sobelYmatrix[i][j] * grayimage[x + i][y + j]))
    elif (case == 2):
        sumFilters = sobelYmatrix[0][1] + sobelYmatrix[0][2] + \
                     sobelYmatrix[1][1] + sobelYmatrix[1][2]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelYmatrix[0][2] * grayimage[x][y]))  # center
        sumFxM += int((sobelYmatrix[0][1] * grayimage[x][y - 1]))  # left1
        sumFxM += int((sobelYmatrix[1][2] * grayimage[x + 1][y]))  # down1
        sumFxM += int((sobelYmatrix[1][1] * grayimage[x + 1][y - 1]))  # down1, left1
    elif (case == 3):
        sumFilters = sobelYmatrix[1][0] + sobelYmatrix[1][1] + \
                     sobelYmatrix[2][0] + sobelYmatrix[2][1]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelYmatrix[2][0] * grayimage[x][y]))  # center
        sumFxM += int((sobelYmatrix[2][1] * grayimage[x][y + 1]))  # right1
        sumFxM += int((sobelYmatrix[1][0] * grayimage[x - 1][y]))  # up1
        sumFxM += int((sobelYmatrix[1][1] * grayimage[x - 1][y + 1]))  # up1, right1
    elif (case == 4):
        sumFilters = sobelYmatrix[1][1] + sobelYmatrix[1][2] + \
                     sobelYmatrix[2][1] + sobelYmatrix[2][2]
        # Iterate a 2x2 grid manually:
        sumFxM += int((sobelYmatrix[2][2] * grayimage[x][y]))  # center
        sumFxM += int((sobelYmatrix[2][1] * grayimage[x][y - 1]))  # left1
        sumFxM += int((sobelYmatrix[1][2] * grayimage[x - 1][y]))  # up1
        sumFxM += int((sobelYmatrix[1][1] * grayimage[x - 1][y - 1]))  # up1, left1
    elif (case == 5):
        '''
        NOTE: Total of Sobel Filter values cancels to 0, so we need a base value.
        '''
        sumFilters = 1
        # Iterate full 3x3 grid manually:
        sumFxM += int((sobelYmatrix[0][0] * grayimage[x - 1][y - 1]))  # up1, left1
        sumFxM += int((sobelYmatrix[0][1] * grayimage[x][y - 1]))  # up1
        sumFxM += int((sobelYmatrix[0][2] * grayimage[x + 1][y - 1]))  # up1, right1
        sumFxM += int((sobelYmatrix[1][0] * grayimage[x - 1][y]))  # left1
        sumFxM += int((sobelYmatrix[1][1] * grayimage[x][y]))  # center
        sumFxM += int((sobelYmatrix[1][2] * grayimage[x + 1][y]))  # right1
        sumFxM += int((sobelYmatrix[2][0] * grayimage[x - 1][y + 1]))  # down1, left1
        sumFxM += int((sobelYmatrix[2][1] * grayimage[x][y + 1]))  # down1
        sumFxM += int((sobelYmatrix[2][2] * grayimage[x + 1][y + 1]))  # down1, right1

    result = int(sumFxM / sumFilters)
    if (result < 0): #flip sign of negative
        result *= -1
    return result


# Define Sobel Gradient Magnitude:
def sobelGradMag(x,y, sobelXtemp, sobelYtemp):
    '''
    :param x: the x location
    :param y: the y location
    :return: result, using equation: sqrt(Gx*Gx + Gy*Gy)
    '''
    result = int(math.sqrt((int(sobelXtemp[x][y])*int(sobelXtemp[x][y]))
                           + (int(sobelYtemp[x][y])*int(sobelYtemp[x][y]))))
    return result

def edgeCheck(i, j, height, width):
    edgeresult = 'N'        #neither by default, middle case
    if (i==0):              #top
        if (j==0):          #topleft
            edgeresult='TLC'
        elif (j==height-1): #topright
            edgeresult='TRC'
        else:
            edgeresult='TE'
    elif (i==width-1):      #bottom
        if (j==0):          #bottomleft
            edgeresult='BLC'
        elif (j==height-1): #bottomright
            edgeresult='BRC'
        else:
            edgeresult='BE'
    elif (j==0):            #left
        edgeresult='LE'
    elif (j==height-1):     #right
        edgeresult='RE'
    return edgeresult

def effectSobel(img):
    height = img.shape[0]
    width = img.shape[1]
    newimg = np.zeros((height, width, 1), np.uint8)

    # Create Sobel variables:
    sobelXmatrix = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.int8)  # Sobel X Filter array
    sobelXtemp = np.zeros((height, width, 1), np.uint8)  # Sobel X image placeholder
    sobelYmatrix = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int8)  # Sobel Y Filter array
    sobelYtemp = np.zeros((height, width, 1), np.uint8)  # Sobel X image placeholder

    print("* Performing Sobel Filter calculations...")
    for i in range(0, width):
        for j in range(0, height):
            # Perform edge/corner checks:
            currentEdge = edgeCheck(i, j)
            # Decide filter action depending on location/case:
            case = 0
            if (currentEdge == 'TLC') or (currentEdge == 'LE') or (currentEdge == 'TE'):
                # Case 1: 2x2 filter, Down+Right
                case = 1
            elif (currentEdge == 'TRC') or (currentEdge == 'RE'):
                # Case 2: 2x2 filter, Down+Left
                case = 2
            elif (currentEdge == 'BLC') or (currentEdge == 'BE'):
                # Case 3: 2x2 filter, Up+Right
                case = 3
            elif (currentEdge == 'BRC'):
                # Case 4: 2x2 filter, Up+Left
                case = 4
            else:
                # Case 5: 3x3 filter, Middle, all directions!
                case = 5

            # Perform Sobel X, Sobel Y, and Gradient Magnitude math,
            # depending on case. Final result goes into final Sobel image.
            sobelXtemp[i][j] = sobelX(case, i, j, sobelXmatrix, newimg)
            sobelYtemp[i][j] = sobelY(case, i, j, sobelYmatrix, newimg)
            newimg[i][j] = sobelGradMag(i, j, sobelXtemp, sobelYtemp)

    return newimg

def effectHarrisBonus(img, img_g):
    #Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = 0.04  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = 11500000000  # Corner Response Threshold; needs to be tweaked per img
    cornerimg = np.copy(img)  # init final corner img, need R channel!
    cornerbonusimg = np.copy(img_g)  # for bonus: visual corner range, in grayscale
    r = 0  # corner response value, will later use equation: det-k(trace^2)
    Ixx = np.zeros((height, width, 1), np.float32)  # hold I-values; NOTE the 32-bit floats needed!
    Ixy = np.zeros((height, width, 1), np.float32)
    Iyy = np.zeros((height, width, 1), np.float32)

    # Built-in Sobel...
    dx = cv2.Sobel(img_g, cv2.CV_64F, 1, 0, ksize=3)  # Hold Sobel derivatives
    dy = cv2.Sobel(img_g, cv2.CV_64F, 0, 1, ksize=3)

    # Get I values from Sobel Gradients:
    for i in range(0, height - 1):
        for j in range(0, width - 1):
            # Fill in I values:
            Ixx[i][j] = dx[i][j] ** 2
            Ixy[i][j] = dy[i][j] * dx[i][j]
            Iyy[i][j] = dy[i][j] ** 2

    #Main Harris loops:
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            # Windowing function:
            windowIxx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]  # multi-shape array
            windowIxy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            windowIyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            # Sum of Squares for windows:
            sumIxx = windowIxx.sum()
            sumIxy = windowIxy.sum()
            sumIyy = windowIyy.sum()

            # Find determinant and trace, use to get corner response
            det = (sumIxx * sumIyy) - (sumIxy ** 2)
            trace = sumIxx + sumIyy
            r = det - (k * (trace ** 2))  # harris r corner response

            # If corner response is over threshold, mark it as valid (red):
            tolerance = 210000000  # for bonus; vicinity of threshold
            if (r > threshold):
                # Immediate pixel
                cornerimg[y][x][0] = 0
                cornerimg[y][x][1] = 0
                cornerimg[y][x][2] = 255
                if (r > (threshold + tolerance)):
                    cornerbonusimg[y][x] = 0  # BONUS
            elif (r > 100) and (r < (threshold - tolerance * 2)):
                cornerbonusimg[y][x] = 255  # BONUS
            elif (r <= 0):
                cornerbonusimg[y][x] = 220  # BONUS

    # Invert the bonus greyscale image to get proper markings:
    cornerbonusimg = cv2.bitwise_not(cornerbonusimg)  # invert the marked grayscale img
    return cornerbonusimg

def effectFullPixelShift(img, offset_range, patchx=0, patchy=0):
    '''
    Leave patchx, patchy at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0] #j, patchy
    width = img.shape[1] #i, patchx
    # newimg = np.zeros((height, width, 3), np.uint8)
    current_offset = 0
    patchx_random = False
    patchy_random = False

    #If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    # Run loop across whole img
    for i in range(0, width):  # NOTE the usage of 1,w-1; 1,h-1
        for j in range(0, height):
            # Determine offset amt and location
            current_offset = random.randrange(-1 * offset_range, offset_range)
            # print("current_offset: " + str(current_offset))
            offset_where = random.randrange(0, 3)
            # print("offset_where: " + str(offset_where))
            # print("--")
            # Produce offset on img pixel locations...
            current_offset_abs = abs(current_offset)
            if (offset_where == 0):  # x
                if ((current_offset_abs + j) < height):
                    newimg[j, i, 0] = img[j + current_offset, i, 0]
                    newimg[j, i, 1] = img[j + current_offset, i, 1]
                    newimg[j, i, 2] = img[j + current_offset, i, 2]
            elif (offset_where == 1):  # y
                if ((current_offset_abs + i) < width):
                    newimg[j, i, 0] = img[j, i + current_offset, 0]
                    newimg[j, i, 1] = img[j, i + current_offset, 1]
                    newimg[j, i, 2] = img[j, i + current_offset, 2]
            elif (offset_where == 2):  # x, y
                if ((current_offset_abs + j) < height):
                    if ((current_offset_abs + i) < width):
                        newimg[j, i, 0] = img[j + current_offset, i + current_offset, 0]
                        newimg[j, i, 1] = img[j + current_offset, i + current_offset, 1]
                        newimg[j, i, 2] = img[j + current_offset, i + current_offset, 2]
    return newimg

def effectRandomPixelShift(img, offset_range, patchx=0, patchy=0, patches=10):
    '''
    Leave patchx, patchy at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0] #j, patchy
    width = img.shape[1] #i, patchx
    # newimg = np.zeros((height, width, 3), np.uint8)
    current_offset = 0
    patchx_random = False
    patchy_random = False

    #If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    # Run loop for 'patches' number of times,
    # if positive, run loop within a patch window,
    # else, run regular loop thru whole img
    if (patches < 10):
        patches = 10
        newimg = np.copy(img)
        for p in range(patches):
            #If patch size is left at 0, randomize!
            if (patchx_random == True):
                patchx = random.randrange(15, int(width // 2))
            if (patchy_random == True):
                patchy = random.randrange(15, int(height // 2))
            istart = random.randrange(0+patchx, width-patchx)
            jstart = random.randrange(0+patchy, height-patchy)
            for i in range(istart, istart + patchx):  # NOTE the usage of 1,w-1; 1,h-1
                for j in range(jstart, jstart + patchy):
                    # Determine offset amt and location
                    current_offset = random.randrange(-1 * offset_range, offset_range)
                    # print("current_offset: " + str(current_offset))
                    offset_where = random.randrange(0, 3)
                    # print("offset_where: " + str(offset_where))
                    # print("--")
                    # Produce offset on img pixel locations...
                    current_offset_abs = abs(current_offset)
                    if (offset_where == 0):  # x
                        if ((current_offset_abs + j) < height):
                            newimg[j, i, 0] = img[j + current_offset, i, 0]
                            newimg[j, i, 1] = img[j + current_offset, i, 1]
                            newimg[j, i, 2] = img[j + current_offset, i, 2]
                    elif (offset_where == 1):  # y
                        if ((current_offset_abs + i) < width):
                            newimg[j, i, 0] = img[j, i + current_offset, 0]
                            newimg[j, i, 1] = img[j, i + current_offset, 1]
                            newimg[j, i, 2] = img[j, i + current_offset, 2]
                    elif (offset_where == 2):  # x, y
                        if ((current_offset_abs + j) < height):
                            if ((current_offset_abs + i) < width):
                                newimg[j, i, 0] = img[j + current_offset, i + current_offset, 0]
                                newimg[j, i, 1] = img[j + current_offset, i + current_offset, 1]
                                newimg[j, i, 2] = img[j + current_offset, i + current_offset, 2]
    return newimg

def effectColorSmear(img, patchx=0, patchy=0, patches=1):
    '''
    Returns a 'smeared' version of img, where given patch has colors copied over
    a set amount.
    Smear from left to right.
    :param img: 
    :param patchx: 
    :param patchy:
    :param patches:
    :return newimg: 
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    #Loop thru patches:
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if (patchx_random == True):
            patchx = random.randrange(15, int(width // 5))
            # print("Current patchx: " + str(patchx))
        if (patchy_random == True):
            patchy = random.randrange(15, int(height // 5))
            # print("Current patchy: " + str(patchx))

        # Get windowing range and ensure no inverse ranges...
        istart = patchx
        if ((width - patchx) > patchx):
            istart = random.randrange(patchx, width - patchx)
        jstart = patchy
        if ((height - patchy) > patchy):
            jstart = random.randrange(patchy, height - patchy)

        #Loop thru window
        for i in range(istart, istart + patchx):
            # Copy initial color (per column):
            colorvalb = img[jstart, i, 0]
            colorvalg = img[jstart, i, 1]
            colorvalr = img[jstart, i, 2]
            for j in range(jstart, jstart + patchy):
                newimg[j, i, 0] = colorvalb
                newimg[j, i, 1] = colorvalg
                newimg[j, i, 2] = colorvalr
    return newimg

def effectColorScratch(img, patchx=0, patchy=0, patches=1, scratchdir=0):
    '''
    Returns a 'scratched' version of img, where given patch has DOMINANT color (BGR) copied over
    a set amount.
    Smear from left to right.
    :param img:
    :param patchx:
    :param patchy:
    :param patches:
    :param scratchdir: 0 for down, 1/else for right
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    #Loop thru patches:
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if (patchx_random == True):
            patchx = random.randrange(15, int(width // 5))
            # print("Current patchx: " + str(patchx))
        if (patchy_random == True):
            patchy = random.randrange(15, int(height // 5))
            # print("Current patchy: " + str(patchx))

        #Get windowing range and ensure no inverse ranges...
        istart = patchx
        if ((width - patchx) > patchx):
            istart = random.randrange(patchx, width - patchx)
        jstart = patchy
        if ((height - patchy) > patchy):
            jstart = random.randrange(patchy, height - patchy)

        #Check scratch direction (0=down, 1=right)
        if (scratchdir == 0):
            #Loop thru window
            for i in range(istart, istart + patchx):
                #Choose amount to scratch over
                scratchamt = random.randrange(5, patchy)
                # Copy initial color (per column):
                colorvalb = img[jstart, i, 0]
                colorvalg = img[jstart, i, 1]
                colorvalr = img[jstart, i, 2]
                # Get max
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(jstart, jstart + scratchamt):
                    # Pick channel
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = colorvalb
                    elif (colormax == colorvalg):
                        newimg[j, i, 1] = colorvalg
                    elif (colormax == colorvalr):
                        newimg[j, i, 2] = colorvalr
        else:
            # Loop thru window
            scratchamt = random.randrange(5, patchx)
            for i in range(istart, istart + scratchamt):
                for j in range(jstart, jstart + patchy):
                    # Choose amount to scratch over
                    scratchamt = random.randrange(5, patchx)
                    # Copy initial color (per column):
                    colorvalb = img[j, istart, 0]  # was jstart, i
                    colorvalg = img[j, istart, 1]
                    colorvalr = img[j, istart, 2]
                    # Get max
                    colormax = max([colorvalb, colorvalg, colorvalr])
                    # Pick channel
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = colorvalb
                    elif (colormax == colorvalg):
                        newimg[j, i, 1] = colorvalg
                    elif (colormax == colorvalr):
                        newimg[j, i, 2] = colorvalr
    return newimg


# Setup Random Call List for effects order
funcNum = 4 #supposed to be at least 6 later
layersNum = random.randrange(60, 200)
effectslist = []
# for e in range(funcNum):
#     effectslist.append((e+1)) #1, 2, 3, etc
for e in range(layersNum):
    effectslist.append(random.randrange(1, funcNum+1))
print(str(effectslist))

#Decide if effect choices will be unique and generate random list:
uniquechoices = random.randrange(0,2)
if (uniquechoices == 0):
    randlist = []
    for e in range(len(effectslist)): #was range(funcNum)
        randlist.append(random.choice(effectslist)) #non-unique choices
else:
    randlist = random.sample(effectslist, len(effectslist)) #unique choices; #was funcNum
print(str(randlist))


# Load Images
'''
IMAGE LIST:
checkerboard.jpg
cityscape_sm.jpg
night_cars.jpg
people_shadows.jpg
butterfly.jpg
redcar3a.png -> edge case, wider pic (1600x429)

BE ABLE to handle diff. dimensions!
'''
print("* Loading initial image...")
# img1 = cv2.imread("images/people_shadows.jpg")
# img1_g = cv2.imread("images/people_shadows.jpg", 0)
img1 = cv2.imread("images/redcar3a.png")
img1_g = cv2.imread("images/redcar3a.png", 0)
# img1 = cv2.imread("images/cityscape_sm.jpg")
# img1_g = cv2.imread("images/cityscape_sm.jpg", 0)

# Perform Functions
def effectCaller(img, effects_order):
    '''
    Calls the effect functions
    :param img: image to use
    :param effects_order: uses randlist to determine order of effects used
    :return newimg:
    '''
    # newimg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    newimg = np.copy(img)
    for e in range(len(effects_order)):
        print("\n* Performing Effect Layer #" + str(e+1) + " / " + str(layersNum))
        print("-- effects_order[e]: " + str(effects_order[e]))
        if (effects_order[e] == 1): #call 1st effect
            pass
            # print("-- Effect: Full Pixel Shift")
            # newimg = effectFullPixelShift(img, int(img.shape[0] // 4), 0, 0)
        elif (effects_order[e] == 2):
            print("-- Effect: Random Pixel Shift")
            patchamt = random.randrange(10, 60)
            newimg = effectRandomPixelShift(img, int(img.shape[0] // 8), 0, 0, patchamt)
        elif (effects_order[e] == 3):
            print("-- Effect: Color Smear")
            patchamt = random.randrange(5, 30)
            newimg = effectColorSmear(img, 0, 0, patchamt)
        elif (effects_order[e] == 4):
            print("-- Effect: Color Scratch")
            patchamt = random.randrange(10, 50)
            scratchdir = random.randrange(0, 2)
            newimg = effectColorScratch(img, 0, 0, patchamt, scratchdir)
        elif (effects_order[e] == 5):
            pass
        elif (effects_order[e] == 6):
            pass
        elif (effects_order[e] == 7):
            pass
        elif (effects_order[e] == 8):
            pass
        elif (effects_order[e] == 9):
            pass
        elif (effects_order[e] == 10):
            pass

    return newimg

# Seed the random library
random.seed()

# Display Results
cv2.imshow("Original Image", img1)

result = effectCaller(img1, randlist)
cv2.imshow("Effect Caller Result", result)
cv2.waitKey(0)

#TEST effectRandomPixelShift
# testimg = effectRandomPixelShift(img1, 50)
# cv2.imshow("Random Pixel Shift - Default", testimg)
# testimg = effectRandomPixelShift(img1, 100, 35, 55, 30)
# cv2.imshow("Random Pixel Shift - Test1", testimg)
# testimg = effectRandomPixelShift(img1, 100, 0, 0, 30)
# cv2.imshow("Random Pixel Shift - Test2 random patches", testimg)
# cv2.waitKey(0)

#TEST colorSmear
# testimg = effectColorSmear(img1, 0, 0, 30)
# # testimg = effectRandomPixelShift(testimg, 50, 0, 0, 20) #composite test
# cv2.imshow("Color Smear - Test1", testimg)
# cv2.waitKey(0)

#TEST colorScratch
# testimg = effectColorScratch(img1, 0, 0, 50)
# testimg2 = effectColorScratch(img1, 0, 0, 50, 1)
# testimg = effectRandomPixelShift(testimg, 50, 0, 0, 20) #composite test
# testimg = effectRandomPixelShift(img1, 50, 0, 0, 20) #composite test
# testimg = effectColorScratch(testimg, 0, 0, 50)
# cv2.imshow("Color Scratch - Test1 Down", testimg)
# cv2.imshow("Color Scratch - Test2 Right", testimg2)
# cv2.waitKey(0)

# cv2.imshow("Glitch Result", result)
# cv2.waitKey(0)
