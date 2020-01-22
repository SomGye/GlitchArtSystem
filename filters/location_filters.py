import numpy as np
import random


def effectRandomPixelShift(img, offset_range=100, patchx=0, patchy=0, patches=4):
    '''
    Leave patchx, patchy at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return newimg:
    '''
    newimg = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    # current_offset = 0
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

    # Run loop for 'patches' number of times,
    # if positive, run loop within a patch window,
    # else, run regular loop thru whole img
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if (patchx_random == True):
            patchx = random.randrange(int(width // 10), int(width // 7))
        if (patchy_random == True):
            patchy = random.randrange(int(height // 13), int(height // 8))
        istart = random.randrange(0 + patchx, width - patchx)
        jstart = random.randrange(0 + patchy, height - patchy)
        for i in range(istart, istart + patchx):  # NOTE the usage of 1,w-1; 1,h-1
            for j in range(jstart, jstart + patchy):
                # Determine if we will shift:
                doShift = random.randrange(0, 3)
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

    # Loop thru patches:
    for p in range(patches):
        # Get patch size, shift amt, and end of patch window
        patchy = random.randrange(5, int(height // 1.2))
        patchysize = random.randrange(10, int(height // 4))
        shift = random.randrange(10, int(width // 5))
        patchend = patchy + patchysize
        if (patchend >= height):  # ensure within size limit
            patchend = height - 1

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

        # Loop thru patch window
        for i in range(width):
            for j in range(patchy, patchend):
                # Get actual shift amount, and determine if overflow:
                shiftresult = i - shift
                if (shiftresult < 0):
                    shiftresult = width + shiftresult
                if (colorfocus == 0):
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = int(img[j, i, 1] // divisor)
                    newimg[j, shiftresult, 2] = int(img[j, i, 2] // divisor)
                elif (colorfocus == 1):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0] // divisor)
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = int(img[j, i, 2] // divisor)
                elif (colorfocus == 2):
                    newimg[j, shiftresult, 0] = int(img[j, i, 0] // divisor)
                    newimg[j, shiftresult, 1] = int(img[j, i, 1] // divisor)
                    newimg[j, shiftresult, 2] = img[j, i, 2]
                else:
                    # ORIG w/o color band
                    newimg[j, shiftresult, 0] = img[j, i, 0]
                    newimg[j, shiftresult, 1] = img[j, i, 1]
                    newimg[j, shiftresult, 2] = img[j, i, 2]
    return newimg
