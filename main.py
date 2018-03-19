import cv2
import numpy as np
import random

#Author: Maxwell Crawford
#CSC475 Final: Glitch Art Generator
#12-7-17 12:27pm
#pres2

## Define Functions
def effectRandomPixelShift(img, offset_range=100, patchx=0, patchy=0, patches=4):
    '''
    Leave patchx, patchy at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0] #j, patchy
    width = img.shape[1] #i, patchx
    # current_offset = 0
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
                # Determine if we will shift:
                doShift = random.randrange(0,3)
                if (doShift == 0):
                    # Determine offset amt and location
                    current_offset = random.randrange(-1 * offset_range, offset_range)
                    offset_where = random.randrange(0, 3)
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
    Create a variable 'sound-wave' by having random up, down, and right
    amounts.
    The wave simulates the switch b/w up and down modes.
    Each channel in the wave shifts the color
     within (-colorshift, colorshift) range.
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
        iend = int(width // 40) #need to tweak
        jend = int(height // 40)
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
    low = 1.07
    med = 1.10
    high = 1.2
    divisor = low

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

        # Get divisor amount:
        lowmedhigh = random.randrange(0, 3)
        if (lowmedhigh == 0):
            divisor = low
        elif (lowmedhigh == 1):
            divisor = med
        elif (lowmedhigh == 2):
            divisor = high

        #Loop thru patch window
        for i in range(width):
            for j in range(patchy, patchend):
                #Get actual shift amount, and determine if overflow:
                shiftresult = i-shift
                if (shiftresult < 0):
                    shiftresult = width+shiftresult
                if (colorfocus == 0):
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = int(img[j, i, 1]//divisor)
                    newimg[j, shiftresult, 2] = int(img[j, i, 2]//divisor)
                elif (colorfocus == 1):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0]//divisor)
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = int(img[j, i, 2]//divisor)
                elif (colorfocus == 2):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0]//divisor)
                    newimg[j, shiftresult, 1] = int(img[j, i, 1]//divisor)
                    newimg[j, shiftresult, 2] = img[j, i, 2]
                else:
                    #ORIG w/o color band
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = img[j, i, 2]
    return newimg

def effectColorCompression(img, patches=4):
    '''
    Color Compression effect: divide color by patch
    depending upon dominant channel chosen
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    if (patches == 4):
        patches = random.randrange(4, 9)

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

def effectHarrisEdgeColorShift(img, img_g):
    '''
    Use Harris Corner Detector to find edges and corners,
    then check if current pixel is on left/right half of image.
    If left half, get dominant color values from pixels to the right by 1/2/3,
    If right half, get dominant color values from pixels to the left by 1/2/3,
    and then heavily attenuate the non-dominant channels.
    :param img:
    :param img_g: greyscale vers. of img
    :return cornerimg:
    '''
    #Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = 0.04  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = random.randrange(9900000000, 11200000000)# Corner Response Threshold; randomize within safe params.
    # threshold = 10000000000  # Corner Response Threshold; needs to be tweaked per img; safe #: 11500000000
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
    for y in range(offset, height - (window_size+1)):
        doHarris = random.randrange(0, 5)
        for x in range(offset, width - (window_size+1)):
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
            if (doHarris == 0):
                if (r < -599000000) or (r > threshold): #test for edges/corners
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
                            cornerimg[y, x + 1, 0] = colorvalb
                            cornerimg[y, x + 1, 1] = colorvalg // 2
                            cornerimg[y, x + 1, 2] = colorvalr // 3
                            #Stripe 2
                            cornerimg[y, x + 2, 0] = colorvalb // 2
                            cornerimg[y, x + 2, 1] = colorvalg
                            cornerimg[y, x + 2, 2] = colorvalr // 2
                            #Stripe 3
                            cornerimg[y, x + 3, 0] = colorvalb // 3
                            cornerimg[y, x + 3, 1] = colorvalg // 2
                            cornerimg[y, x + 3, 2] = colorvalr
                            cornerimg[y, x + 4, 0] = colorvalb // 3
                            cornerimg[y, x + 4, 1] = colorvalg // 2
                            cornerimg[y, x + 4, 2] = colorvalr
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
                            cornerimg[y, x + 4, 0] = colorvalb
                            cornerimg[y, x + 4, 1] = colorvalg // 2
                            cornerimg[y, x + 4, 2] = colorvalr // 3
                        elif (colormax == colorvalr):
                            #Stripe 1
                            cornerimg[y, x + 1, 0] = colorvalb // 3
                            cornerimg[y, x + 1, 1] = colorvalg // 2
                            cornerimg[y, x + 1, 2] = colorvalr
                            #Stripe 2
                            cornerimg[y, x + 2, 0] = colorvalb
                            cornerimg[y, x + 2, 1] = colorvalg // 2
                            cornerimg[y, x + 2, 2] = colorvalr // 3
                            #Stripe 3
                            cornerimg[y, x + 3, 0] = colorvalb // 2
                            cornerimg[y, x + 3, 1] = colorvalg
                            cornerimg[y, x + 3, 2] = colorvalr // 2
                            cornerimg[y, x + 4, 0] = colorvalb // 2
                            cornerimg[y, x + 4, 1] = colorvalg
                            cornerimg[y, x + 4, 2] = colorvalr // 2
                    elif (x>(width//2)):
                        # Pick channel
                        if (colormax == colorvalb):
                            # Stripe 1
                            cornerimg[y, x - 1, 0] = colorvalb
                            cornerimg[y, x - 1, 1] = colorvalg // 2
                            cornerimg[y, x - 1, 2] = colorvalr // 3
                            # Stripe 2
                            cornerimg[y, x - 2, 0] = colorvalb // 2
                            cornerimg[y, x - 2, 1] = colorvalg
                            cornerimg[y, x - 2, 2] = colorvalr // 2
                            # Stripe 3
                            cornerimg[y, x - 3, 0] = colorvalb // 3
                            cornerimg[y, x - 3, 1] = colorvalg // 2
                            cornerimg[y, x - 3, 2] = colorvalr
                            cornerimg[y, x - 4, 0] = colorvalb // 3
                            cornerimg[y, x - 4, 1] = colorvalg // 2
                            cornerimg[y, x - 4, 2] = colorvalr
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
                            cornerimg[y, x - 4, 0] = colorvalb
                            cornerimg[y, x - 4, 1] = colorvalg // 2
                            cornerimg[y, x - 4, 2] = colorvalr // 3
                        elif (colormax == colorvalr):
                            # Stripe 1
                            cornerimg[y, x - 1, 0] = colorvalb // 3
                            cornerimg[y, x - 1, 1] = colorvalg // 2
                            cornerimg[y, x - 1, 2] = colorvalr
                            # Stripe 2
                            cornerimg[y, x - 2, 0] = colorvalb
                            cornerimg[y, x - 2, 1] = colorvalg // 2
                            cornerimg[y, x - 2, 2] = colorvalr // 3
                            # Stripe 3
                            cornerimg[y, x - 3, 0] = colorvalb // 2
                            cornerimg[y, x - 3, 1] = colorvalg
                            cornerimg[y, x - 3, 2] = colorvalr // 2
                            cornerimg[y, x - 4, 0] = colorvalb // 2
                            cornerimg[y, x - 4, 1] = colorvalg
                            cornerimg[y, x - 4, 2] = colorvalr // 2
    return cornerimg

def copyOver(img, orig_img, option=0):
        '''
        Copy over original image parts over an edited image.
        Option 0: Random
        Option 1: Top
        Option 2: Bottom
        Option 3: left
        Option 4: right
        Option 5: Top left
        Option 6: Top right
        Option 7: Bottom left
        Option 8: Bottom right
        else, do nothing!
        :param img:
        :param orig_img:
        :param option:
        :return newimg:
        '''
        newimg = np.copy(img)
        height = img.shape[0]  # j, patchy
        width = img.shape[1]  # i, patchx

        # Choose action depending on option
        if (option == 0):
            option = random.randrange(1, 11)  # 9,10 = nothing!
        if (option == 1):  # top half
            for i in range(width):
                for j in range(height // 2):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 2):  # bottom half
            for i in range(width):
                for j in range(height // 2, height):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 3):  # left half
            for i in range(width // 2):
                for j in range(height):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 4):  # right half
            for i in range(width // 2, width):
                for j in range(height):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 5):  # top left
            for i in range(width // 2):
                for j in range(height // 2):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 6):  # top right
            for i in range(width // 2, width):
                for j in range(height // 2):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 7):  # bottom left
            for i in range(width // 2):
                for j in range(height // 2, height):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 8):  # bottom right
            for i in range(width // 2, width):
                for j in range(height // 2, height):
                    newimg[j, i, 0] = orig_img[j, i, 0]
                    newimg[j, i, 1] = orig_img[j, i, 1]
                    newimg[j, i, 2] = orig_img[j, i, 2]
        return newimg

def effectConvolutionEdgeLines(img, patches=2):
    '''
    Use a 3x3 convolution kernel to detect horizontal lines,
    then within a patch of (istop-istart, jstop-jstart) determine
    whether to convolve a particular color channel or all of them evenly, per j.
    :param img:
    :param patches: optional # of patches
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    convoMatrix = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]]) #horiz lines
    sumFilters = 1 #needed to ensure stability with all imgs

    for p in range(patches):
        doAny = random.randrange(0, 4)
        if (doAny == 0):
            continue
        # Determine whether to have patch size in orig. style, or new style
        doStyle = random.randrange(0, 2)
        istart = 0  # init. for scope
        istop = 0
        jstart = 0
        jstop = 0
        if (doStyle == 0):  # orig. window
            istart = random.randrange(int(width // 2.5))
            istop = random.randrange(istart, int(width // 1.7))
            jstart = random.randrange(int(height // 2.5))
            jstop = random.randrange(jstart, int(height // 1.7))
        else:
            istart = random.randrange(int(width // 11), int(width // 2.2))  # range around the middle
            istop = random.randrange(int(width // 1.8), int(width // 1.05))
            jstart = random.randrange(int(height // 11), int(height // 2.2))
            jstop = random.randrange(int(height // 1.8), int(height // 1.05))
        for i in range(istart, istop):
            doAny = random.randrange(0, 4)
            if (doAny == 0):
                continue
            for j in range(jstart, jstop):
                doAllChannels = random.randrange(0,8) #if 0, pick 1 channel, else do all channels; was 0,8
                sumFxM = 0
                if (doAllChannels == 0):
                    c = random.randrange(0, 3)
                    sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                    sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sumFxM / sumFilters))
                    newimg[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                        sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sumFxM/sumFilters))
                        newimg[j,i,c] = result
    return newimg

def effectConvolutionEdgeDilation(img, patches=2):
    '''
    Use Edge Detection Convolution kernel, then dilate the result,
     allowing purposeful integer underflow.
    :param img:
    :param patches: optional # of patches
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    convoMatrix = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])  # edge detect

    for p in range(patches):
        doAny = random.randrange(0, 4)
        if (doAny == 0):
            continue
        # Determine whether to have patch size in orig. style, or new style
        doStyle = random.randrange(0, 2)
        istart = 0  # init. for scope
        istop = 0
        jstart = 0
        jstop = 0
        if (doStyle == 0):  # orig. window
            istart = random.randrange(int(width // 2.5))
            istop = random.randrange(istart, int(width // 1.7))
            jstart = random.randrange(int(height // 2.5))
            jstop = random.randrange(jstart, int(height // 1.7))
        else:
            istart = random.randrange(int(width // 11), int(width // 2.2))  # range around the middle
            istop = random.randrange(int(width // 1.8), int(width // 1.05))
            jstart = random.randrange(int(height // 11), int(height // 2.2))
            jstop = random.randrange(int(height // 1.8), int(height // 1.05))
        for i in range(istart, istop):
            doAny = random.randrange(0, 4)
            if (doAny == 0):
                continue
            for j in range(jstart, jstop):
                doAllChannels = random.randrange(0,8) #if 0, pick 1 channel, else do all channels; was 0,8
                sumFxM = 0
                sumFilters = random.randrange(5, 90) #dilate result for underflow later
                if (doAllChannels == 0):
                    c = random.randrange(0, 3)
                    sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                    sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sumFxM / sumFilters))
                    #Trim the results above 0 -> reduce overall darkness
                    if (result <= 0):
                        newimg[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                        sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sumFxM/sumFilters))
                        # Trim the results above 0 -> reduce overall darkness
                        if (result <= 0):
                            newimg[j, i, c] = result
    return newimg

def effectConvolutionDynamic(img, patches=2):
    '''
    Generate a different Convolution Kernel each time!
    :param img:
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    convoMatrix = None #init as null
    randomEach = random.randrange(0,5)
    #If randomEach false, just generate kernel once!
    if (randomEach == 0):
        convoMatrix = np.array([[random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                [random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                [random.randrange(-2, 3), random.randrange(-2, 3),
                                 random.randrange(-2, 3)]])  # DYNAMIC, -2->2

    for p in range(patches):
        #If randomEach is true, generate convolution kernel EACH PATCH
        if (randomEach != 0):
            convoMatrix = np.array([[random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                    [random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                    [random.randrange(-2, 3), random.randrange(-2, 3),
                                     random.randrange(-2, 3)]])  # DYNAMIC, -2->2
        doAny = random.randrange(0, 4) #randomly allow null patch
        if (doAny == 0):
            continue

        #Determine whether to have patch size in orig. style, or new style
        doStyle = random.randrange(0,2)
        istart = 0 #init. for scope
        istop = 0
        jstart = 0
        jstop = 0
        if (doStyle == 0): #orig. window
            istart = random.randrange(int(width // 3.5))
            istop = random.randrange(istart, int(width // 2.7))
            jstart = random.randrange(int(height // 3.5))
            jstop = random.randrange(jstart, int(height // 2.7))
        else:
            istart = random.randrange(int(width // 11), int(width // 2.2))  # range around the middle
            istop = random.randrange(int(width // 1.8), int(width // 1.05))
            jstart = random.randrange(int(height // 11), int(height // 2.2))
            jstop = random.randrange(int(height // 1.8), int(height // 1.05))
        for i in range(istart, istop):
            doAny = random.randrange(0, 4) #randomly allow null patch
            if (doAny == 0):
                continue
            for j in range(jstart, jstop):
                doAllChannels = random.randrange(0,8) #if 0, pick 1 channel, else do all channels; was 0,8
                sumFxM = 0
                sumFilters = random.randrange(1, 25)
                if (doAllChannels == 0):
                    c = random.randrange(0, 3)
                    sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                    sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sumFxM / sumFilters))
                    newimg[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sumFxM += int((convoMatrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sumFxM += int((convoMatrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sumFxM += int((convoMatrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sumFxM += int((convoMatrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sumFxM += int((convoMatrix[1][1] * int(img[j][i][c])))  # center
                        sumFxM += int((convoMatrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sumFxM += int((convoMatrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sumFxM += int((convoMatrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sumFxM += int((convoMatrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sumFxM/sumFilters))
                        newimg[j, i, c] = result
    return newimg

def effectCrossHatch(img, randomspots=150):
    '''
    Random hatch spots in variable X patterns.
    Similar to static.
    :param img:
    :return newimg:
    '''
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    newimg = np.copy(img)
    maxcolor = 255

    # Loop
    if (randomspots == 150):
        randomspots = random.randrange(150, 301)
    for r in range(randomspots):
        randj = random.randrange(10, height - 10)
        randi = random.randrange(10, width - 10)
        spotcolor = random.randrange(int(maxcolor // 8), int(maxcolor // 1.2))

        newimg[randj, randi, 0] = spotcolor
        newimg[randj, randi, 1] = spotcolor
        newimg[randj, randi, 2] = spotcolor
        # Begin series of 8 dots for Cross pattern,
        # with each spot being optional for variation
        doSpot = random.randrange(0,3)
        if (doSpot != 0):
            newimg[randj - 1, randi - 1, 0] = spotcolor
            newimg[randj - 1, randi - 1, 1] = spotcolor
            newimg[randj - 1, randi - 1, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj + 1, randi + 1, 0] = spotcolor
            newimg[randj + 1, randi + 1, 1] = spotcolor
            newimg[randj + 1, randi + 1, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj - 1, randi + 1, 0] = spotcolor
            newimg[randj - 1, randi + 1, 1] = spotcolor
            newimg[randj - 1, randi + 1, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj + 1, randi - 1, 0] = spotcolor
            newimg[randj + 1, randi - 1, 1] = spotcolor
            newimg[randj + 1, randi - 1, 2] = spotcolor
        #--
        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj - 3, randi - 3, 0] = spotcolor
            newimg[randj - 3, randi - 3, 1] = spotcolor
            newimg[randj - 3, randi - 3, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj + 3, randi + 3, 0] = spotcolor
            newimg[randj + 3, randi + 3, 1] = spotcolor
            newimg[randj + 3, randi + 3, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj - 3, randi + 3, 0] = spotcolor
            newimg[randj - 3, randi + 3, 1] = spotcolor
            newimg[randj - 3, randi + 3, 2] = spotcolor

        doSpot = random.randrange(0, 3)
        if (doSpot != 0):
            newimg[randj + 3, randi - 3, 0] = spotcolor
            newimg[randj + 3, randi - 3, 1] = spotcolor
            newimg[randj + 3, randi - 3, 2] = spotcolor
    return newimg

def copyOverColorDistort(img, orig_img, option=0):
        '''
        Copy over dominant color columns from the orig. image parts
         over an edited image.
        The non-dominant channels are attenuated.
        Option 0: Random
        Option 1: Top
        Option 2: Bottom
        Option 3: left
        Option 4: right
        Option 5: Top left
        Option 6: Top right
        Option 7: Bottom left
        Option 8: Bottom right
        else, do nothing!
        :param img:
        :param orig_img:
        :param option:
        :return newimg:
        '''
        newimg = np.copy(img)
        height = img.shape[0]  # j, patchy
        width = img.shape[1]  # i, patchx

        # Choose action depending on option
        if (option == 0):
            option = random.randrange(1, 11)  # 9,10 = nothing!
        if (option == 1):  # top half
            for i in range(width):
                # Copy initial color:
                colorvalb = img[0, i, 0]
                colorvalg = img[0, i, 1]
                colorvalr = img[0, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 2):  # bottom half
            for i in range(width):
                # Copy initial color:
                colorvalb = img[height // 2, i, 0]
                colorvalg = img[height // 2, i, 1]
                colorvalr = img[height // 2, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2, height):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 3):  # left half
            for i in range(width // 2):
                # Copy initial color:
                colorvalb = img[0, i, 0]
                colorvalg = img[0, i, 1]
                colorvalr = img[0, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 4):  # right half
            for i in range(width // 2, width):
                # Copy initial color:
                colorvalb = img[0, i, 0]
                colorvalg = img[0, i, 1]
                colorvalr = img[0, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 5):  # top left
            for i in range(width // 2):
                # Copy initial color:
                colorvalb = img[0, i, 0]
                colorvalg = img[0, i, 1]
                colorvalr = img[0, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 6):  # top right
            for i in range(width // 2, width):
                # Copy initial color:
                colorvalb = img[0, i, 0]
                colorvalg = img[0, i, 1]
                colorvalr = img[0, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 7):  # bottom left
            for i in range(width // 2):
                # Copy initial color:
                colorvalb = img[height // 2, i, 0]
                colorvalg = img[height // 2, i, 1]
                colorvalr = img[height // 2, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2, height):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        elif (option == 8):  # bottom right
            for i in range(width // 2, width):
                # Copy initial color:
                colorvalb = img[height // 2, i, 0]
                colorvalg = img[height // 2, i, 1]
                colorvalr = img[height // 2, i, 2]
                # Determine dominant color and half others:
                colormax = max([colorvalb, colorvalg, colorvalr])
                for j in range(height // 2, height):
                    if (colormax == colorvalb):
                        newimg[j, i, 0] = orig_img[j, i, 0]
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2] // 2
                    elif (colormax == colorvalg):
                        newimg[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                        newimg[j, i, 1] = orig_img[j, i, 1]
                        newimg[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                    else:
                        newimg[j, i, 0] = orig_img[j, i, 0] // 2
                        newimg[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                        newimg[j, i, 2] = orig_img[j, i, 2]
        return newimg
## END FUNCTIONS
## BEGIN CALLS/SETUP

# Setup Random Call List for effects order
funcNum = 15 #currently supported effects...
layersNum = random.randrange(5, 26)
effectslist = []
print("* Generating Random Order of Effects...")
for e in range(layersNum):
    # Generate effect layer #:
    effect = random.randrange(1, funcNum+1)
    effectslist.append(effect) #add resulting effect to list

#Generate random list from chosen effects,
# ensuring two conditions:
randlist = []
for e in range(len(effectslist)):
    choice = random.choice(effectslist)
    #1) Prevent CopyOver (10) from being 1st effect!
    if (e == 0):
        while True:
            if (choice == 10):
                choice = random.randrange(1, funcNum + 1)
                continue
            else:
                break
    else:
        #2) If effect is same as last one, try again:
        while True:
            if (choice == randlist[-1]):
                choice = random.randrange(1, funcNum + 1)
                continue
            else:
                break
    randlist.append(choice)  # non-unique choices after conditions

# Load Images;
# 1) Check test.png (in images folder or not)
# 2) Check test.jpg (in images folder or not)
# 3) Else, check other defined images...
print("* Loading initial image...")
img1 = cv2.imread("images/test.png")
img1_g = cv2.imread("images/test.png", 0)
if (img1 is None):
    img1 = cv2.imread("images/test.jpg")
    img1_g = cv2.imread("images/test.jpg", 0)
if (img1 is None):
    img1 = cv2.imread("test.png")
    img1_g = cv2.imread("test.png", 0)
if (img1 is None):
    img1 = cv2.imread("test.jpg")
    img1_g = cv2.imread("test.jpg", 0)
if (img1 is None):
    # MY PREDEFINED IMAGES...
    # img1 = cv2.imread("images/people_shadows.jpg")
    # img1_g = cv2.imread("images/people_shadows.jpg", 0)
    # img1 = cv2.imread("images/redcar3a.png")
    # img1_g = cv2.imread("images/redcar3a.png", 0)
    # img1 = cv2.imread("images/testglitch.jpg")
    # img1_g = cv2.imread("images/testglitch.jpg", 0)
    # img1 = cv2.imread("images/checkerboard.jpg")
    # img1_g = cv2.imread("images/checkerboard.jpg", 0)
    # img1 = cv2.imread("images/night_cars.jpg")
    # img1_g = cv2.imread("images/night_cars.jpg", 0)
    # img1 = cv2.imread("images/blugrad.jpg")
    # img1_g = cv2.imread("images/blugrad.jpg", 0)
    # img1 = cv2.imread("images/pexels (1).jpeg")
    # img1_g = cv2.imread("images/pexels (1).jpeg", 0)
    # img1 = cv2.imread("images/pexels (2).jpeg")
    # img1_g = cv2.imread("images/pexels (2).jpeg", 0)
    # img1 = cv2.imread("images/pexels (3).jpeg")
    # img1_g = cv2.imread("images/pexels (3).jpeg", 0)
    img1 = cv2.imread("images/pexels (4).jpeg")
    img1_g = cv2.imread("images/pexels (4).jpeg", 0)
    # img1 = cv2.imread("images/pexels (5).jpeg")
    # img1_g = cv2.imread("images/pexels (5).jpeg", 0)
    # img1 = cv2.imread("images/lidar1.png")
    # img1_g = cv2.imread("images/lidar1.png", 0)

# Perform Functions
def effectCaller2(img, effect):
    '''
    Calls the effect functions and displays basic info.
    This takes a copy of the existing layered image and
    returns an image with a new effect layer applied.
    :param img: image to use
    :param effects_order: uses randlist to determine order of effects used
    :return newimg:
    '''
    newimg = np.copy(img)
    newimg_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (effect == 1): #call 1st effect
        print("-- Effect: Random Pixel Shift")
        newimg = effectRandomPixelShift(img)
    elif (effect == 2):
        print("-- Effect: Color Smear")
        newimg = effectColorSmear(img)
    elif (effect == 3):
        print("-- Effect: Color Scratch")
        scratchdir = random.randrange(0, 2)
        newimg = effectColorScratch(img, 0, 0, 4, scratchdir)
    elif (effect == 4):
        print("-- Effect: SoundWave")
        newimg = effectSoundWave(img)
    elif (effect == 5):
        print("-- Effect: Static")
        newimg = effectStatic(img)
    elif (effect == 6):
        print("-- Effect: Scanlines")
        newimg = effectScanlines(img)
    elif (effect == 7):
        print("-- Effect: Horiz. Shift")
        newimg = effectHorizShift(img)
    elif (effect == 8):
        print("-- Effect: Color Compression Bands")
        newimg = effectColorCompression(img)
    elif (effect == 9):
        print("-- Effect: Harris Color Shift")
        newimg = effectHarrisEdgeColorShift(img, newimg_g)
    elif (effect == 10):
        print("-- Effect: Copy Over (Original Vers.)")
        newimg = copyOver(img, img1)
    elif (effect == 11):
        print("-- Effect: Convolution Edge Lines")
        newimg = effectConvolutionEdgeLines(img)
    elif (effect == 12):
        print("-- Effect: Convolution Edge Dilation")
        newimg = effectConvolutionEdgeDilation(img)
    elif (effect == 13):
        print("-- Effect: Convolution Dynamics")
        newimg = effectConvolutionDynamic(img)
    elif (effect == 14):
        print("-- Effect: Cross Hatch")
        newimg = effectCrossHatch(img)
    elif (effect == 15):
        print("-- Effect: Copy Over (Color Distort)")
        newimg = copyOverColorDistort(img, img1)
    return newimg

# Seed the random library
random.seed()
## END SETUP
# Display Results
large = 1600
result = np.copy(img1)
for e in range(len(randlist)):
    print("\n* Performing Effect Layer #" + str(e+1) + " / " + str(layersNum))
    result = effectCaller2(result, randlist[e])
cv2.imwrite("resulttest.png", result, [cv2.IMWRITE_PNG_COMPRESSION,0]) #note the PNG, lowest compression!
cv2.imshow("Original Image", img1)
if (img1.shape[0] > large): #If image is 'large', then resize for convenience
    result = cv2.resize(result, (int(0.5*result.shape[1]), int(0.5*result.shape[0])), interpolation=cv2.INTER_AREA)
cv2.imshow("Effect Caller2 Result", result)
cv2.waitKey(0)