import cv2
import numpy as np
import random


def effect_color_smear(img, patchx=0, patchy=0, patches=5):
    """
    Returns a 'smeared' version of img, where given patch has colors copied over
    a set amount.
    Smear from left to right.
    :param img:
    :param patchx:
    :param patchy:
    :param patches:
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    min_patches = 5
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if patchx == 0:
        patchx_random = True
    if patchy == 0:
        patchy_random = True

    if patches < min_patches:
        patches = min_patches

    # Loop thru patches:
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if patchx_random:
            patchx = random.randrange(10, int(width // 7))
            # print("Current patchx: " + str(patchx))
        if patchy_random:
            patchy = random.randrange(10, int(height // 7))
            # print("Current patchy: " + str(patchx))

        # Get windowing range and ensure no inverse ranges...
        istart = patchx
        if (width - patchx) > patchx:
            istart = random.randrange(patchx, width - patchx)
        jstart = patchy
        if (height - patchy) > patchy:
            jstart = random.randrange(patchy, height - patchy)

        # Loop thru window
        for i in range(istart, istart + patchx):
            # Copy initial color (per column):
            color_val_b = img[jstart, i, 0]
            color_val_g = img[jstart, i, 1]
            color_val_r = img[jstart, i, 2]
            for j in range(jstart, jstart + patchy):
                new_img[j, i, 0] = color_val_b
                new_img[j, i, 1] = color_val_g
                new_img[j, i, 2] = color_val_r
    return new_img


def effect_color_scratch(img, patchx=0, patchy=0, patches=4, scratchdir=0):
    """
    Returns a 'scratched' version of img, where given patch has DOMINANT color (BGR) copied over
    a set amount.
    Smear from left to right.
    :param img:
    :param patchx:
    :param patchy:
    :param patches:
    :param scratchdir: 0 for down, 1/else for right
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    min_patches = 4
    patchx_random = False
    patchy_random = False

    # If patch sizes are left a 0, randomize!
    if patchx == 0:
        patchx_random = True
    if patchy == 0:
        patchy_random = True

    if patches < min_patches:
        patches = min_patches

    # Loop thru patches:
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if patchx_random:
            patchx = random.randrange(15, int(width // 5))
            # print("Current patchx: " + str(patchx))
        if patchy_random:
            patchy = random.randrange(15, int(height // 5))
            # print("Current patchy: " + str(patchx))

        # Get windowing range and ensure no inverse ranges...
        istart = patchx
        if (width - patchx) > patchx:
            istart = random.randrange(patchx, width - patchx)
        jstart = patchy
        if (height - patchy) > patchy:
            jstart = random.randrange(patchy, height - patchy)

        # Check scratch direction (0=down, 1=right)
        if scratchdir == 0:
            # Loop thru window
            for i in range(istart, istart + patchx):
                # Choose amount to scratch over
                scratchamt = random.randrange(5, patchy)
                # Copy initial color (per column):
                color_val_b = img[jstart, i, 0]
                color_val_g = img[jstart, i, 1]
                color_val_r = img[jstart, i, 2]
                # Get max
                color_max = max([color_val_b, color_val_g, color_val_r])
                for j in range(jstart, jstart + scratchamt):
                    # Pick channel
                    if (color_max == color_val_b):
                        new_img[j, i, 0] = color_val_b
                    elif (color_max == color_val_g):
                        new_img[j, i, 1] = color_val_g
                    elif (color_max == color_val_r):
                        new_img[j, i, 2] = color_val_r
        else:
            # Loop thru window
            scratchamt = random.randrange(5, patchx)
            for i in range(istart, istart + scratchamt):
                for j in range(jstart, jstart + patchy):
                    # Choose amount to scratch over
                    scratchamt = random.randrange(5, patchx)
                    # Copy initial color (per column):
                    color_val_b = img[j, istart, 0]  # was jstart, i
                    color_val_g = img[j, istart, 1]
                    color_val_r = img[j, istart, 2]
                    # Get max
                    color_max = max([color_val_b, color_val_g, color_val_r])
                    # Pick channel
                    if (color_max == color_val_b):
                        new_img[j, i, 0] = color_val_b
                    elif (color_max == color_val_g):
                        new_img[j, i, 1] = color_val_g
                    elif (color_max == color_val_r):
                        new_img[j, i, 2] = color_val_r
    return new_img


def effect_color_compression(img, patches=4):
    """
    Color Compression effect: divide color by patch
    depending upon dominant channel chosen
    :param img:
    :param patches:
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    if patches == 4:
        patches = random.randrange(4, 9)

    # Loop thru patches:
    for p in range(patches):
        # Get patch size, shift amt, and end of patch window
        patchy = random.randrange(5, int(height // 1.2))
        patchysize = random.randrange(5, int(height // 12))
        patchend = patchy + patchysize

        if patchend >= height:  # ensure within size limit
            patchend = height - 1

        # Loop thru patch window
        for i in range(width):
            for j in range(patchy, patchend):
                # Randomize color channel to focus on:
                colorfocus = random.randrange(0, 3)
                if colorfocus == 0:
                    new_img[j, i, 0] = img[j, i, 0]
                    new_img[j, i, 1] = int(img[j, i, 1] // 1.4)
                    new_img[j, i, 2] = int(img[j, i, 2] // 1.4)
                elif colorfocus == 1:
                    new_img[j, i, 0] = int(img[j, i, 0] // 1.4)
                    new_img[j, i, 1] = img[j, i, 1]
                    new_img[j, i, 2] = int(img[j, i, 2] // 1.4)
                elif colorfocus == 2:
                    new_img[j, i, 0] = int(img[j, i, 0] // 1.4)
                    new_img[j, i, 1] = int(img[j, i, 1] // 1.4)
                    new_img[j, i, 2] = img[j, i, 2]
    return new_img


# Define Harris Color Shift Stripe solution:
def harris_color_shift_striping(img, corner_img, x, y, width):
    """
    Subsidiary function to do color attenuation in "stripes",
     dependent on dominant color channel, for Harris Color Shift.
    :param img:
    :param corner_img:
    :param x:
    :param y:
    :param width:
    :return corner_img:
    """
    # Copy initial color:
    color_val_b = img[y, x, 0]
    color_val_g = img[y, x, 1]
    color_val_r = img[y, x, 2]
    # Get max
    color_max = max([color_val_b, color_val_g, color_val_r])
    # Check if left/right side of half
    if x < (width // 2):
        # Pick channel
        if color_max == color_val_b:
            # Stripe 1
            corner_img[y, x + 1, 0] = color_val_b
            corner_img[y, x + 1, 1] = color_val_g // 2
            corner_img[y, x + 1, 2] = color_val_r // 3
            # Stripe 2
            corner_img[y, x + 2, 0] = color_val_b // 2
            corner_img[y, x + 2, 1] = color_val_g
            corner_img[y, x + 2, 2] = color_val_r // 2
            # Stripe 3
            corner_img[y, x + 3, 0] = color_val_b // 3
            corner_img[y, x + 3, 1] = color_val_g // 2
            corner_img[y, x + 3, 2] = color_val_r
            corner_img[y, x + 4, 0] = color_val_b // 3
            corner_img[y, x + 4, 1] = color_val_g // 2
            corner_img[y, x + 4, 2] = color_val_r
        elif color_max == color_val_g:
            # Stripe 1
            corner_img[y, x + 1, 0] = color_val_b // 2
            corner_img[y, x + 1, 1] = color_val_g
            corner_img[y, x + 1, 2] = color_val_r // 2
            # Stripe 2
            corner_img[y, x + 2, 0] = color_val_b // 3
            corner_img[y, x + 2, 1] = color_val_g // 2
            corner_img[y, x + 2, 2] = color_val_r
            # Stripe 3
            corner_img[y, x + 3, 0] = color_val_b
            corner_img[y, x + 3, 1] = color_val_g // 2
            corner_img[y, x + 3, 2] = color_val_r // 3
            corner_img[y, x + 4, 0] = color_val_b
            corner_img[y, x + 4, 1] = color_val_g // 2
            corner_img[y, x + 4, 2] = color_val_r // 3
        elif color_max == color_val_r:
            # Stripe 1
            corner_img[y, x + 1, 0] = color_val_b // 3
            corner_img[y, x + 1, 1] = color_val_g // 2
            corner_img[y, x + 1, 2] = color_val_r
            # Stripe 2
            corner_img[y, x + 2, 0] = color_val_b
            corner_img[y, x + 2, 1] = color_val_g // 2
            corner_img[y, x + 2, 2] = color_val_r // 3
            # Stripe 3
            corner_img[y, x + 3, 0] = color_val_b // 2
            corner_img[y, x + 3, 1] = color_val_g
            corner_img[y, x + 3, 2] = color_val_r // 2
            corner_img[y, x + 4, 0] = color_val_b // 2
            corner_img[y, x + 4, 1] = color_val_g
            corner_img[y, x + 4, 2] = color_val_r // 2
    elif x > (width // 2):
        # Pick channel
        if color_max == color_val_b:
            # Stripe 1
            corner_img[y, x - 1, 0] = color_val_b
            corner_img[y, x - 1, 1] = color_val_g // 2
            corner_img[y, x - 1, 2] = color_val_r // 3
            # Stripe 2
            corner_img[y, x - 2, 0] = color_val_b // 2
            corner_img[y, x - 2, 1] = color_val_g
            corner_img[y, x - 2, 2] = color_val_r // 2
            # Stripe 3
            corner_img[y, x - 3, 0] = color_val_b // 3
            corner_img[y, x - 3, 1] = color_val_g // 2
            corner_img[y, x - 3, 2] = color_val_r
            corner_img[y, x - 4, 0] = color_val_b // 3
            corner_img[y, x - 4, 1] = color_val_g // 2
            corner_img[y, x - 4, 2] = color_val_r
        elif color_max == color_val_g:
            # Stripe 1
            corner_img[y, x - 1, 0] = color_val_b // 2
            corner_img[y, x - 1, 1] = color_val_g
            corner_img[y, x - 1, 2] = color_val_r // 2
            # Stripe 2
            corner_img[y, x - 2, 0] = color_val_b // 3
            corner_img[y, x - 2, 1] = color_val_g // 2
            corner_img[y, x - 2, 2] = color_val_r
            # Stripe 3
            corner_img[y, x - 3, 0] = color_val_b
            corner_img[y, x - 3, 1] = color_val_g // 2
            corner_img[y, x - 3, 2] = color_val_r // 3
            corner_img[y, x - 4, 0] = color_val_b
            corner_img[y, x - 4, 1] = color_val_g // 2
            corner_img[y, x - 4, 2] = color_val_r // 3
        elif color_max == color_val_r:
            # Stripe 1
            corner_img[y, x - 1, 0] = color_val_b // 3
            corner_img[y, x - 1, 1] = color_val_g // 2
            corner_img[y, x - 1, 2] = color_val_r
            # Stripe 2
            corner_img[y, x - 2, 0] = color_val_b
            corner_img[y, x - 2, 1] = color_val_g // 2
            corner_img[y, x - 2, 2] = color_val_r // 3
            # Stripe 3
            corner_img[y, x - 3, 0] = color_val_b // 2
            corner_img[y, x - 3, 1] = color_val_g
            corner_img[y, x - 3, 2] = color_val_r // 2
            corner_img[y, x - 4, 0] = color_val_b // 2
            corner_img[y, x - 4, 1] = color_val_g
            corner_img[y, x - 4, 2] = color_val_r // 2
    return corner_img


def effect_harris_edge_color_shift(img, img_g):
    """
    Use Harris Corner Detector to find edges and corners,
    then check if current pixel is on left/right half of image.
    If left half, get dominant color values from pixels to the right by 1/2/3,
    If right half, get dominant color values from pixels to the left by 1/2/3,
    and then heavily attenuate the non-dominant channels.
    :param img:
    :param img_g: greyscale vers. of img
    :return corner_img:
    """
    # Define variables
    height = img.shape[0]
    width = img.shape[1]
    window_size = 3  # Windowing Size
    offset = window_size // 2  # Offset is 1/2window size in each direction
    k = random.randrange(4, 8) * 0.01  # Harris Corner constant; usually b/w 0.04-0.07
    threshold = random.randrange(9900000000, 11200000000)  # Corner Response Threshold; randomize within safe params.
    corner_img = np.copy(img)  # init final corner img, need R channel!
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

    # Main Harris loops:
    for y in range(offset, height - (window_size + 1)):
        do_harris = random.randrange(0, 5)
        for x in range(offset, width - (window_size + 1)):
            # Windowing function:
            window_ixx = Ixx[y - offset:y + offset + 1, x - offset:x + offset + 1]  # multi-shape array
            window_ixy = Ixy[y - offset:y + offset + 1, x - offset:x + offset + 1]
            window_iyy = Iyy[y - offset:y + offset + 1, x - offset:x + offset + 1]

            # Sum of Squares for windows:
            sum_ixx = window_ixx.sum()
            sum_ixy = window_ixy.sum()
            sum_iyy = window_iyy.sum()

            # Find determinant and trace, use to get corner response
            det = (sum_ixx * sum_iyy) - (sum_ixy ** 2)
            trace = sum_ixx + sum_iyy
            r = det - (k * (trace ** 2))  # harris r corner response

            # If corner response is over threshold, mark it as valid (red):
            if do_harris == 0:
                if (r < -599000000) or (r > threshold):  # test for edges/corners
                    # Perform Harris Color Shift, in stripes:
                    corner_img = harris_color_shift_striping(img, corner_img, x, y, width)
    return corner_img


def effect_convolution_edge_lines(img, patches=2):
    """
    Use a 3x3 convolution kernel to detect horizontal lines,
    then within a patch of (istop-istart, jstop-jstart) determine
    whether to convolve a particular color channel or all of them evenly, per j.
    :param img:
    :param patches: optional # of patches
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    convo_matrix = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])  # horiz lines
    sum_filters = 1  # needed to ensure stability with all imgs

    for p in range(patches):
        do_any = random.randrange(0, 4)
        if do_any == 0:
            continue
        # Determine whether to have patch size in orig. style, or new style
        do_style = random.randrange(0, 2)
        if do_style == 0:  # orig. window
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
            do_any = random.randrange(0, 4)
            if do_any == 0:
                continue
            for j in range(jstart, jstop):
                do_all_channels = random.randrange(0, 8)  # if 0, pick 1 channel, else do all channels; was 0,8
                sum_FxM = 0
                if do_all_channels == 0:
                    c = random.randrange(0, 3)
                    sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                    sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sum_FxM / sum_filters))
                    new_img[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                        sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sum_FxM / sum_filters))
                        new_img[j, i, c] = result
    return new_img


def effect_convolution_edge_dilation(img, patches=2):
    """
    Use Edge Detection Convolution kernel, then dilate the result,
     allowing purposeful integer underflow.
    :param img:
    :param patches: optional # of patches
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    convo_matrix = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]])  # edge detect

    for p in range(patches):
        do_any = random.randrange(0, 4)
        if do_any == 0:
            continue
        # Determine whether to have patch size in orig. style, or new style
        do_style = random.randrange(0, 2)
        if do_style == 0:  # orig. window
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
            do_any = random.randrange(0, 4)
            if do_any == 0:
                continue
            for j in range(jstart, jstop):
                do_all_channels = random.randrange(0, 8)  # if 0, pick 1 channel, else do all channels; was 0,8
                sum_FxM = 0
                sum_filters = random.randrange(5, 90)  # dilate result for underflow later
                if do_all_channels == 0:
                    c = random.randrange(0, 3)
                    sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                    sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sum_FxM / sum_filters))
                    # Trim the results above 0 -> reduce overall darkness
                    if result <= 0:
                        new_img[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                        sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sum_FxM / sum_filters))
                        # Trim the results above 0 -> reduce overall darkness
                        if result <= 0:
                            new_img[j, i, c] = result
    return new_img


def effect_convolution_dynamic(img, patches=2):
    """
    Generate a different Convolution Kernel each time!
    :param img:
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    convo_matrix = None  # init as null
    random_each = random.randrange(0, 5)
    # If random_each false, just generate kernel once!
    if random_each == 0:
        convo_matrix = np.array([[random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                 [random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                 [random.randrange(-2, 3), random.randrange(-2, 3),
                                  random.randrange(-2, 3)]])  # DYNAMIC, -2->2

    for p in range(patches):
        # If random_each is true, generate convolution kernel EACH PATCH
        if random_each != 0:
            convo_matrix = np.array([[random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                     [random.randrange(-2, 3), random.randrange(-2, 3), random.randrange(-2, 3)],
                                     [random.randrange(-2, 3), random.randrange(-2, 3),
                                      random.randrange(-2, 3)]])  # DYNAMIC, -2->2
        do_any = random.randrange(0, 4)  # randomly allow null patch
        if do_any == 0:
            continue

        # Determine whether to have patch size in orig. style, or new style
        do_style = random.randrange(0, 2)
        if do_style == 0:  # orig. window
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
            do_any = random.randrange(0, 4)  # randomly allow null patch
            if (do_any == 0):
                continue
            for j in range(jstart, jstop):
                do_all_channels = random.randrange(0, 8)  # if 0, pick 1 channel, else do all channels; was 0,8
                sum_FxM = 0
                sum_filters = random.randrange(1, 25)
                if do_all_channels == 0:
                    c = random.randrange(0, 3)
                    sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                    sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                    sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                    sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                    sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                    sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                    sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                    sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                    sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                    result = int((sum_FxM / sum_filters))
                    new_img[j, i, c] = result
                else:
                    for c in range(0, 3):
                        sum_FxM += int((convo_matrix[0][0] * int(img[j - 1][i - 1][c])))  # up1, left1
                        sum_FxM += int((convo_matrix[0][1] * int(img[j][i - 1][c])))  # up1
                        sum_FxM += int((convo_matrix[0][2] * int(img[j + 1][i - 1][c])))  # up1, right1
                        sum_FxM += int((convo_matrix[1][0] * int(img[j - 1][i][c])))  # left1
                        sum_FxM += int((convo_matrix[1][1] * int(img[j][i][c])))  # center
                        sum_FxM += int((convo_matrix[1][2] * int(img[j + 1][i][c])))  # right1
                        sum_FxM += int((convo_matrix[2][0] * int(img[j - 1][i + 1][c])))  # down1, left1
                        sum_FxM += int((convo_matrix[2][1] * int(img[j][i + 1][c])))  # down1
                        sum_FxM += int((convo_matrix[2][2] * int(img[j + 1][i + 1][c])))  # down1, right1
                        result = int((sum_FxM / sum_filters))
                        new_img[j, i, c] = result
    return new_img
