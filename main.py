import cv2
import numpy as np
import random
import math

#Author: Maxwell Crawford
#CSC475 Final: Glitch Art Generator
#11-12-17 9:47pm
#harris start

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

def effectHarrisEdges(img, img_g):
    #Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = 0.04  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = 11500000000  # Corner Response Threshold; needs to be tweaked per img
    cornerimg = np.copy(img)  # init final corner img, need R channel!
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
            # if (r > threshold):
            if (r < -599000000): #TEST for edges
                # Immediate pixel
                cornerimg[y][x][0] = 0
                cornerimg[y][x][1] = 0
                cornerimg[y][x][2] = 255
    # TEST return orig corner
    return cornerimg

def effectHarrisCorners(img, img_g):
    #Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = 0.04  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = 11500000000  # Corner Response Threshold; needs to be tweaked per img
    cornerimg = np.copy(img)  # init final corner img, need R channel!
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
            if (r > threshold): #TEST for corners
            # if (r < -50000000): #TEST for edges
                # Immediate pixel
                cornerimg[y][x][0] = 0
                cornerimg[y][x][1] = 0
                cornerimg[y][x][2] = 255
    # TEST return orig corner
    return cornerimg
##-- end old

## HARRIS EXPERIMENT
def effectHarrisEdgeColorShift(img, img_g):
    #Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = 0.04  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = 11500000000  # Corner Response Threshold; needs to be tweaked per img
    cornerimg = np.copy(img)  # init final corner img, need R channel!
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
    for y in range(offset, height - 4): #was offset, height-offset; offset, width-offset;
        for x in range(offset, width - 4):
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
            # if (r > threshold):
            if (r < -599000000) or (r > threshold): #TEST for edges OR corners...
                ## PERFORM ACTIONS HERE...
                # Copy initial color:
                colorvalb = img[y, x, 0]
                colorvalg = img[y, x, 1]
                colorvalr = img[y, x, 2]
                # Get max
                colormax = max([colorvalb, colorvalg, colorvalr])
                #Check if left/right side of half
                if (x<(width//2)):
                    # Pick channel
                    if (colormax == colorvalb):
                        #Stripe 1
                        cornerimg[y, x - 1, 0] = colorvalb
                        cornerimg[y, x - 1, 1] = colorvalg // 2
                        cornerimg[y, x - 1, 2] = colorvalr // 3
                        #Stripe 2
                        cornerimg[y, x - 2, 0] = colorvalb // 2
                        cornerimg[y, x - 2, 1] = colorvalg
                        cornerimg[y, x - 2, 2] = colorvalr // 2
                        #Stripe 3
                        cornerimg[y, x - 3, 0] = colorvalb // 3
                        cornerimg[y, x - 3, 1] = colorvalg // 2
                        cornerimg[y, x - 3, 2] = colorvalr
                    elif (colormax == colorvalg):
                        # Stripe 1
                        cornerimg[y, x - 1, 0] = colorvalb // 2
                        cornerimg[y, x - 1, 1] = colorvalg
                        cornerimg[y, x - 1, 2] = colorvalr // 2
                        # Stripe 2
                        cornerimg[y, x - 2, 0] = colorvalb // 3
                        cornerimg[y, x - 2, 1] = colorvalg // 2
                        cornerimg[y, x - 2, 2] = colorvalr
                        # Stripe 3
                        cornerimg[y, x - 3, 0] = colorvalb
                        cornerimg[y, x - 3, 1] = colorvalg // 2
                        cornerimg[y, x - 3, 2] = colorvalr // 3
                    elif (colormax == colorvalr):
                        cornerimg[y, x, 2] = colorvalr
                        #Stripe 1
                        cornerimg[y, x - 1, 0] = colorvalb // 3
                        cornerimg[y, x - 1, 1] = colorvalg // 2
                        cornerimg[y, x - 1, 2] = colorvalr
                        #Stripe 2
                        cornerimg[y, x - 2, 0] = colorvalb
                        cornerimg[y, x - 2, 1] = colorvalg // 2
                        cornerimg[y, x - 2, 2] = colorvalr // 3
                        #Stripe 3
                        cornerimg[y, x - 3, 0] = colorvalb // 2
                        cornerimg[y, x - 3, 1] = colorvalg
                        cornerimg[y, x - 3, 2] = colorvalr // 2
                elif (x>(width//2)):
                    # Pick channel
                    if (colormax == colorvalb):
                        # Stripe 1
                        cornerimg[y, x + 1, 0] = colorvalb
                        cornerimg[y, x + 1, 1] = colorvalg // 2
                        cornerimg[y, x + 1, 2] = colorvalr // 3
                        # Stripe 2
                        cornerimg[y, x + 2, 0] = colorvalb // 2
                        cornerimg[y, x + 2, 1] = colorvalg
                        cornerimg[y, x + 2, 2] = colorvalr // 2
                        # Stripe 3
                        cornerimg[y, x + 3, 0] = colorvalb // 3
                        cornerimg[y, x + 3, 1] = colorvalg // 2
                        cornerimg[y, x + 3, 2] = colorvalr
                    elif (colormax == colorvalg):
                        # Stripe 1
                        cornerimg[y, x + 1, 0] = colorvalb // 2
                        cornerimg[y, x + 1, 1] = colorvalg
                        cornerimg[y, x + 1, 2] = colorvalr // 2
                        # Stripe 2
                        cornerimg[y, x + 2, 0] = colorvalb // 3
                        cornerimg[y, x + 2, 1] = colorvalg // 2
                        cornerimg[y, x + 2, 2] = colorvalr
                        # Stripe 3
                        cornerimg[y, x + 3, 0] = colorvalb
                        cornerimg[y, x + 3, 1] = colorvalg // 2
                        cornerimg[y, x + 3, 2] = colorvalr // 3
                    elif (colormax == colorvalr):
                        cornerimg[y, x, 2] = colorvalr
                        # Stripe 1
                        cornerimg[y, x + 1, 0] = colorvalb // 3
                        cornerimg[y, x + 1, 1] = colorvalg // 2
                        cornerimg[y, x + 1, 2] = colorvalr
                        # Stripe 2
                        cornerimg[y, x + 2, 0] = colorvalb
                        cornerimg[y, x + 2, 1] = colorvalg // 2
                        cornerimg[y, x + 2, 2] = colorvalr // 3
                        # Stripe 3
                        cornerimg[y, x + 3, 0] = colorvalb // 2
                        cornerimg[y, x + 3, 1] = colorvalg
                        cornerimg[y, x + 3, 2] = colorvalr // 2
    return cornerimg
## --end harris

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

def effectRandomPixelShift(img, offset_range, patchx=0, patchy=0, patches=4):
    '''
    Leave patchx, patchy at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0] #j, patchy
    width = img.shape[1] #i, patchx
    current_offset = 0
    min_patches = 4
    patchx_random = False
    patchy_random = False

    #If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    if (patches < min_patches):
        patches = min_patches

    # Run loop for 'patches' number of times,
    # if positive, run loop within a patch window,
    # else, run regular loop thru whole img
    for p in range(patches):
        #If patch size is left at 0, randomize!
        if (patchx_random == True):
            patchx = random.randrange(int(width // 10), int(width // 7))
        if (patchy_random == True):
            patchy = random.randrange(int(height // 13), int(height // 8))
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

def effectColorSmear(img, patchx=0, patchy=0, patches=5):
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
    min_patches = 5
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    if (patches < min_patches):
        patches = min_patches

    #Loop thru patches:
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if (patchx_random == True):
            patchx = random.randrange(10, int(width // 7))
            # print("Current patchx: " + str(patchx))
        if (patchy_random == True):
            patchy = random.randrange(10, int(height // 7))
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

def effectColorScratch(img, patchx=0, patchy=0, patches=4, scratchdir=0):
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
    min_patches = 4
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if (patchx == 0):
        patchx_random = True
    if (patchy == 0):
        patchy_random = True

    if (patches < min_patches):
        patches = min_patches

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

def effectSoundWave(img, colorshift=25):
    '''

    :param img:
    :return:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    halfheight = int(height // 2)
    newimg = np.copy(img)

    #Choose init wave params
    updown = random.randrange(0,2)
    upamt = random.randrange(10, int(halfheight // 1.2))
    downamt = random.randrange(10, int(halfheight // 1.2))
    rightamt = random.randrange(1,11)
    for i in range(width):
        if (i % rightamt == 0): #switch dir
            #Reinit. wave params
            updown = random.randrange(0, 2)
            upamt = random.randrange(10, int(halfheight // 1.2))
            downamt = random.randrange(10, int(halfheight // 1.2))
            rightamt = random.randrange(1, 11)
        if (updown == 0): #go up
            for j in range(halfheight, halfheight-upamt, -1):
                #Randomize color
                newb = img[j][i][0] + (random.randrange(-colorshift, colorshift))
                newg = img[j][i][1] + (random.randrange(-colorshift, colorshift))
                newr = img[j][i][2] + (random.randrange(-colorshift, colorshift))
                #Apply color
                newimg[j][i][0] = newb
                newimg[j][i][1] = newg
                newimg[j][i][2] = newr
        elif (updown == 1): #go down
            for j in range(halfheight, halfheight+downamt):
                # Randomize color
                newb = img[j][i][0] + (random.randrange(-colorshift, colorshift))
                newg = img[j][i][1] + (random.randrange(-colorshift, colorshift))
                newr = img[j][i][2] + (random.randrange(-colorshift, colorshift))
                # Apply color
                newimg[j][i][0] = newb
                newimg[j][i][1] = newg
                newimg[j][i][2] = newr

        # for j in range(height):
    return newimg
def effectStatic(img):
    '''
    Static effect: randomize 'pock' marks of random greyscale values;
    Loop 1: cover whole image with semi-uniform specks of random color
    Loop 2: choose random spots and fill those in too
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    maxcolor = 255

    #Loop 1
    for i in range(width):
        iend = int(width // 90)
        jend = int(height // 90)
        if (iend <= 3):
            iend = 6
        if (jend <= 3):
            jend = 6
        spacingi = random.randrange(3, iend) # was 60,60
        spacingj = random.randrange(3, jend)
        for j in range(height):
            if ((i % spacingi) == 0) and ((j % spacingj) == 0):
            # if (i % spacingi == 0): #solid vert. lines
                useColor = random.randrange(0, 4)
                if (useColor == 0):
                    newimg[j, i, 0] = img[j, i, 0] // 3
                    newimg[j, i, 1] = img[j, i, 1] // 3
                    newimg[j, i, 2] = img[j, i, 2] // 3
                else:
                    spotcolor = random.randrange(int(maxcolor//7), int(maxcolor//1.5)) #roughly 42->159
                    newimg[j, i, 0] = spotcolor
                    newimg[j, i, 1] = spotcolor
                    newimg[j, i, 2] = spotcolor
    #Loop 2
    randomspots = random.randrange(90,180)
    for r in range(randomspots):
        useSquare = random.randrange(0, 3) #0=regular, 1-2=use square patch
        randj = random.randrange(5, height-5)
        randi = random.randrange(5, width-5)
        spotcolor = random.randrange(int(maxcolor // 7), int(maxcolor // 1.5))  # roughly 42->159
        if (useSquare == 0):
            newimg[randj, randi, 0] = spotcolor
            newimg[randj, randi, 1] = spotcolor
            newimg[randj, randi, 2] = spotcolor
        else:
            newimg[randj, randi, 0] = spotcolor
            newimg[randj, randi, 1] = spotcolor
            newimg[randj, randi, 2] = spotcolor

            newimg[randj - 1, randi - 1, 0] = spotcolor
            newimg[randj - 1, randi - 1, 1] = spotcolor
            newimg[randj - 1, randi - 1, 2] = spotcolor

            newimg[randj + 1, randi + 1, 0] = spotcolor
            newimg[randj + 1, randi + 1, 1] = spotcolor
            newimg[randj + 1, randi + 1, 2] = spotcolor

            newimg[randj - 1, randi + 1, 0] = spotcolor
            newimg[randj - 1, randi + 1, 1] = spotcolor
            newimg[randj - 1, randi + 1, 2] = spotcolor

            newimg[randj + 1, randi - 1, 0] = spotcolor
            newimg[randj + 1, randi - 1, 1] = spotcolor
            newimg[randj + 1, randi - 1, 2] = spotcolor
    return newimg

def effectScanlines(img, lines=5):
    '''
    Scanlines effect: horizontal lines of solid grayscale static,
    with bursts of color
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    maxcolor = 255
    if (lines <= 5):
        lines = random.randrange(5, 25)
    for l in range(lines):
        randomj = random.randrange(int(height // 12), int(height // 1.2))
        for i in range(width):
            spacingj = random.randrange(int(height // 12), int(height // 1.2))
            for j in range(height):
                if ((j % spacingj) == 0):
                    useColor = random.randrange(0,4) #determine if g.s. or not
                    useThickLine = random.randrange(0,4) #determine if thicker or not
                    if (useColor == 0):
                        # Copy initial color:
                        colorvalb = img[randomj, i, 0]
                        colorvalg = img[randomj, i, 1]
                        colorvalr = img[randomj, i, 2]
                        # Determine dominant color and half others:
                        colormax = max([colorvalb, colorvalg, colorvalr])
                        if (colormax == colorvalb):
                            newimg[randomj, i, 0] = colorvalb
                            newimg[randomj, i, 1] = colorvalg//2
                            newimg[randomj, i, 2] = colorvalr//2
                            if (useThickLine != 0):
                                newimg[randomj+1, i, 0] = colorvalb
                                newimg[randomj+1, i, 1] = colorvalg // 2
                                newimg[randomj+1, i, 2] = colorvalr // 2
                        elif (colormax == colorvalg):
                            newimg[randomj, i, 0] = colorvalb // 2
                            newimg[randomj, i, 1] = colorvalg
                            newimg[randomj, i, 2] = colorvalr // 2
                            if (useThickLine != 0):
                                newimg[randomj+1, i, 0] = colorvalb // 2
                                newimg[randomj+1, i, 1] = colorvalg
                                newimg[randomj+1, i, 2] = colorvalr // 2
                        elif (colormax == colorvalr):
                            newimg[randomj, i, 0] = colorvalb // 2
                            newimg[randomj, i, 1] = colorvalg // 2
                            newimg[randomj, i, 2] = colorvalr
                            if (useThickLine != 0):
                                newimg[randomj+1, i, 0] = colorvalb // 2
                                newimg[randomj+1, i, 1] = colorvalg // 2
                                newimg[randomj+1, i, 2] = colorvalr
                    else:
                        spotcolor = random.randrange(int(maxcolor//6), int(maxcolor//1.6)) #roughly 42->159
                        newimg[randomj, i, 0] = spotcolor #was j,i
                        newimg[randomj, i, 1] = spotcolor
                        newimg[randomj, i, 2] = spotcolor
                        if (useThickLine != 0):
                            newimg[randomj+1, i, 0] = spotcolor  # was j,i
                            newimg[randomj+1, i, 1] = spotcolor
                            newimg[randomj+1, i, 2] = spotcolor
    return newimg

def effectHorizShift(img):
    '''
    Horizontal Shift effect: take entire set of rows and shift over,
    take entire selected row patch (y1 to y2) left;
    Increase offset each shift, but if overflow-> kick over to right side (width-current)
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    patches = random.randrange(3, 7)

    #Loop thru patches:
    for p in range(patches):
        #Get patch size, shift amt, and end of patch window
        patchy = random.randrange(5, int(height // 1.2))
        patchysize = random.randrange(10, int(height // 4))
        shift = random.randrange(10, int(width // 5))
        patchend = patchy + patchysize
        if (patchend >= height): #ensure within size limit
            patchend = height-1

        # Randomize color channel to focus on:
        colorfocus = random.randrange(0, 4)

        #Loop thru patch window
        for i in range(width):
            for j in range(patchy, patchend):
                #Get actual shift amount, and determine if overflow:
                shiftresult = i-shift
                if (shiftresult < 0):
                    shiftresult = width+shiftresult
                if (colorfocus == 0):
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = int(img[j, i, 1]//1.2)
                    newimg[j, shiftresult, 2] = int(img[j, i, 2]//1.2)
                elif (colorfocus == 1):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0]//1.2)
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = int(img[j, i, 2]//1.2)
                elif (colorfocus == 2):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0]//1.2)
                    newimg[j, shiftresult, 1] = int(img[j, i, 1]//1.2)
                    newimg[j, shiftresult, 2] = img[j, i, 2]
                else:
                    #ORIG w/o color band
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = img[j, i, 2]
    return newimg

def effectColorCompression(img):
    '''
    Color Compression effect: divide color by patch
    depending upon dominant channel chosen
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    patches = random.randrange(4, 8)

    #Loop thru patches:
    for p in range(patches):
        #Get patch size, shift amt, and end of patch window
        patchy = random.randrange(5, int(height // 1.2))
        patchysize = random.randrange(5, int(height // 12))
        patchend = patchy + patchysize

        if (patchend >= height): #ensure within size limit
            patchend = height-1

        #Loop thru patch window
        for i in range(width):
            for j in range(patchy, patchend):
                # Randomize color channel to focus on:
                colorfocus = random.randrange(0, 3)
                if (colorfocus == 0):
                    newimg[j, i, 0] = img[j, i, 0]
                    newimg[j, i, 1] = int(img[j, i, 1]//1.4)
                    newimg[j, i, 2] = int(img[j, i, 2]//1.4)
                elif (colorfocus == 1):
                    newimg[j, i, 0] = int(img[j, i, 0]//1.4)
                    newimg[j, i, 1] = img[j, i, 1]
                    newimg[j, i, 2] = int(img[j, i, 2]//1.4)
                elif (colorfocus == 2):
                    newimg[j, i, 0] = int(img[j, i, 0]//1.4)
                    newimg[j, i, 1] = int(img[j, i, 1]//1.4)
                    newimg[j, i, 2] = img[j, i, 2]
    return newimg

def effectCrossHatch(img):
    '''

    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    maxcolor = 255

    # Loop
    randomspots = random.randrange(120, 240)
    for r in range(randomspots):
        randj = random.randrange(10, height - 10)
        randi = random.randrange(10, width - 10)
        spotcolor = random.randrange(int(maxcolor // 7), int(maxcolor // 1.5))  # roughly 42->159

        newimg[randj, randi, 0] = spotcolor
        newimg[randj, randi, 1] = spotcolor
        newimg[randj, randi, 2] = spotcolor
        #--

        newimg[randj - 1, randi - 1, 0] = spotcolor
        newimg[randj - 1, randi - 1, 1] = spotcolor
        newimg[randj - 1, randi - 1, 2] = spotcolor

        newimg[randj + 1, randi + 1, 0] = spotcolor
        newimg[randj + 1, randi + 1, 1] = spotcolor
        newimg[randj + 1, randi + 1, 2] = spotcolor

        newimg[randj - 1, randi + 1, 0] = spotcolor
        newimg[randj - 1, randi + 1, 1] = spotcolor
        newimg[randj - 1, randi + 1, 2] = spotcolor

        newimg[randj + 1, randi - 1, 0] = spotcolor
        newimg[randj + 1, randi - 1, 1] = spotcolor
        newimg[randj + 1, randi - 1, 2] = spotcolor
        #--

        newimg[randj - 3, randi - 3, 0] = spotcolor
        newimg[randj - 3, randi - 3, 1] = spotcolor
        newimg[randj - 3, randi - 3, 2] = spotcolor

        newimg[randj + 3, randi + 3, 0] = spotcolor
        newimg[randj + 3, randi + 3, 1] = spotcolor
        newimg[randj + 3, randi + 3, 2] = spotcolor

        newimg[randj - 3, randi + 3, 0] = spotcolor
        newimg[randj - 3, randi + 3, 1] = spotcolor
        newimg[randj - 3, randi + 3, 2] = spotcolor

        newimg[randj + 3, randi - 3, 0] = spotcolor
        newimg[randj + 3, randi - 3, 1] = spotcolor
        newimg[randj + 3, randi - 3, 2] = spotcolor
    return newimg
## END FUNCTIONS

# Setup Random Call List for effects order
funcNum = 8 #currently supported effects...
layersNum = random.randrange(15, 50) #was 15,60
effectslist = []
# for e in range(funcNum):
#     effectslist.append((e+1)) #1, 2, 3, etc
for e in range(layersNum):
    effectslist.append(random.randrange(1, funcNum+1))
# print(str(effectslist))

#Decide if effect choices will be unique and generate random list:
uniquechoices = random.randrange(0,2)
if (uniquechoices == 0):
    randlist = []
    for e in range(len(effectslist)): #was range(funcNum)
        randlist.append(random.choice(effectslist)) #non-unique choices
else:
    randlist = random.sample(effectslist, len(effectslist)) #unique choices; #was funcNum
print("* Effects Call Order: ")
print(str(randlist))


# Load Images
'''
IMAGE LIST:
checkerboard.jpg (great)
cityscape_sm.jpg
night_cars.jpg (not as good)
people_shadows.jpg
butterfly.jpg
redcar3a.png -> edge case, wider pic (1600x429), great
testglitch.jpg (good)
blugrad.jpg
warptower1.jpg
--
pexels (1).jpeg
pexels (2).jpeg
pexels (3).jpeg
pexels (4).jpeg
pexels (5).jpeg
'''
print("* Loading initial image...")
# img1 = cv2.imread("images/people_shadows.jpg")
# img1_g = cv2.imread("images/people_shadows.jpg", 0)
img1 = cv2.imread("images/redcar3a.png")
img1_g = cv2.imread("images/redcar3a.png", 0)
# img1 = cv2.imread("images/cityscape_sm.jpg")
# img1_g = cv2.imread("images/cityscape_sm.jpg", 0)
# img1 = cv2.imread("images/testglitch.jpg")
# img1_g = cv2.imread("images/testglitch.jpg", 0)
# img1 = cv2.imread("images/checkerboard.jpg")
# img1_g = cv2.imread("images/checkerboard.jpg", 0)
# img1 = cv2.imread("images/night_cars.jpg")
# img1_g = cv2.imread("images/night_cars.jpg", 0)
# img1 = cv2.imread("images/blugrad.jpg")
# img1_g = cv2.imread("images/blugrad.jpg", 0)
# img1 = cv2.imread("images/warptower1.jpg")
# img1_g = cv2.imread("images/warptower1.jpg", 0)
# img1 = cv2.imread("images/pexels (1).jpeg")
# img1_g = cv2.imread("images/pexels (1).jpeg", 0)
# img1 = cv2.imread("images/pexels (2).jpeg")
# img1_g = cv2.imread("images/pexels (2).jpeg", 0)
# img1 = cv2.imread("images/pexels (3).jpeg")
# img1_g = cv2.imread("images/pexels (3).jpeg", 0)
# img1 = cv2.imread("images/pexels (4).jpeg")
# img1_g = cv2.imread("images/pexels (4).jpeg", 0)
# img1 = cv2.imread("images/pexels (5).jpeg")
# img1_g = cv2.imread("images/pexels (5).jpeg", 0)

# Perform Functions
def effectCaller2(img, effect):
    '''
    Calls the effect functions
    :param img: image to use
    :param effects_order: uses randlist to determine order of effects used
    :return newimg:
    '''
    newimg = np.copy(img)
    if (effect == 1): #call 1st effect
        print("-- Effect 1: Random Pixel Shift")
        # patchamt = random.randrange(10, 60)
        # newimg = effectRandomPixelShift(img, int(img.shape[0] // 8), 0, 0, patchamt)
        newimg = effectRandomPixelShift(img, 50)
    elif (effect == 2):
        print("-- Effect 2: Color Smear")
        # patchamt = random.randrange(5, 30)
        # newimg = effectColorSmear(img, 0, 0, patchamt)
        newimg = effectColorSmear(img)
    elif (effect == 3):
        print("-- Effect 3: Color Scratch")
        # patchamt = random.randrange(10, 50)
        scratchdir = random.randrange(0, 2)
        # newimg = effectColorScratch(img, 0, 0, patchamt, scratchdir)
        newimg = effectColorScratch(img, 0, 0, 4, scratchdir)
    elif (effect == 4):
        print("-- Effect 4: SoundWave")
        newimg = effectSoundWave(img)
    elif (effect == 5):
        print("-- Effect 5: Static")
        newimg = effectStatic(img)
    elif (effect == 6):
        print("-- Effect 6: Scanlines")
        newimg = effectScanlines(img)
    elif (effect == 7):
        print("-- Effect 7: Horiz. Shift")
        newimg = effectHorizShift(img)
    elif (effect == 8):
        print("-- Effect 8: Color Compression Bands")
        newimg = effectColorCompression(img)
    elif (effect == 9):
        pass
    elif (effect == 10):
        pass

    return newimg

# Seed the random library
random.seed()

# Display Results

#effectCaller
# result = np.copy(img1)
# result = effectCaller(result, randlist) #was effectCaller(img1...)
# cv2.imshow("Effect Caller Result", result)
# cv2.waitKey(0)

#effectCaller2
# result = np.copy(img1)
# for e in range(len(randlist)):
#     print("\n* Performing Effect Layer #" + str(e+1) + " / " + str(layersNum))
    # print("-- randlist[e]: " + str(randlist[e]))
    # result = effectCaller2(result, randlist[e])
# cv2.imwrite("resulttest.png", result, [cv2.IMWRITE_PNG_COMPRESSION,0]) #note the PNG, lowest compression!
# cv2.imshow("Original Image", img1)
# cv2.imshow("Effect Caller2 Result", result)
# cv2.waitKey(0)

#TEST effectRandomPixelShift
# testimg = effectRandomPixelShift(img1, 50)
# cv2.imwrite("randompxshift1.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Random Pixel Shift - Default", testimg)

# testimg = effectRandomPixelShift(img1, 100, 0, 0, 30)
# cv2.imwrite("randompxshift.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Random Pixel Shift - Test2 random patches", testimg)
# cv2.waitKey(0)

#TEST colorSmear
# testimg = effectColorSmear(img1, 0, 0, 25)
# cv2.imwrite("colorsmear.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Color Smear - Test1", testimg)
# cv2.waitKey(0)

#TEST colorScratch
# testimg = effectColorScratch(img1, 0, 0, 50)
# testimg2 = effectColorScratch(img1, 0, 0, 50, 1)

# cv2.imwrite("colorscratch.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imwrite("colorscratch.png", testimg2, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Color Scratch - Test1 Down", testimg)
# cv2.imshow("Color Scratch - Test2 Right", testimg2)
# cv2.waitKey(0)

#TEST effectSoundWave
# testimg = effectSoundWave(img1)
# cv2.imwrite("soundwave.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("SoundWave1", testimg)
# cv2.waitKey(0)

#TEST effectStatic
# testimg = effectStatic(img1)
# cv2.imwrite("static.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Static1", testimg)
# cv2.waitKey(0)

#TEST effectScanlines
# testimg = effectScanlines(img1, 25) #manual 25 lines
# cv2.imwrite("scanlines.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Scanlines1", testimg)
# cv2.waitKey(0)

#TEST effectHorizShift
# testimg = effectHorizShift(img1)
# cv2.imwrite("horizshift.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("HorizShift1", testimg)
# cv2.waitKey(0)

#TEST effectColorCompression
# testimg = effectColorCompression(img1)
# cv2.imwrite("colorcomp.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("ColorCompression1", testimg)
# cv2.waitKey(0)

#TEST Harris'
# testimg = effectHarrisBonus(img1, img1_g)
# testimg2 = effectHarrisCorners(img1, img1_g)
# testimg3 = effectHarrisEdges(img1, img1_g)
testimg4 = effectHarrisEdgeColorShift(img1, img1_g)
# cv2.imwrite("harrisbonus.png", testimg, [cv2.IMWRITE_PNG_COMPRESSION,0])
# cv2.imshow("Harris Bonus", testimg)
# cv2.imshow("Harris Corners", testimg2)
# cv2.imshow("Harris Edges", testimg3)
cv2.imshow("Harris Edge Color Shift", testimg4)
cv2.waitKey(0)