import numpy as np
import random


def effect_random_pixel_shift(img, offset_range=100, patch_x=0, patch_y=0, patches=4):
    """
    Leave patch_x, patch_y at 0 to randomize!
    :param img:
    :param offset_range: max amount it can be shifted
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patch_y
    width = img.shape[1]  # i, patch_x
    # current_offset = 0
    min_patches = 4
    patch_x_random = False
    patch_y_random = False

    # If patch sizes are left a 0, randomize!
    if patch_x == 0:
        patch_x_random = True
    if patch_y == 0:
        patch_y_random = True

    if patches < min_patches:
        patches = min_patches

    # Run loop for 'patches' number of times,
    # if positive, run loop within a patch window,
    # else, run regular loop thru whole img
    for p in range(patches):
        # If patch size is left at 0, randomize!
        if patch_x_random:
            patch_x = random.randrange(int(width // 10), int(width // 7))
        if patch_y_random:
            patch_y = random.randrange(int(height // 13), int(height // 8))
        istart = random.randrange(0 + patch_x, width - patch_x)
        jstart = random.randrange(0 + patch_y, height - patch_y)
        for i in range(istart, istart + patch_x):  # NOTE the usage of 1,w-1; 1,h-1
            for j in range(jstart, jstart + patch_y):
                # Determine if we will shift:
                do_shift = random.randrange(0, 3)
                if do_shift == 0:
                    # Determine offset amt and location
                    current_offset = random.randrange(-1 * offset_range, offset_range)
                    offset_where = random.randrange(0, 3)
                    # Produce offset on img pixel locations...
                    current_offset_abs = abs(current_offset)
                    if offset_where == 0:  # x
                        if (current_offset_abs + j) < height:
                            new_img[j, i, 0] = img[j + current_offset, i, 0]
                            new_img[j, i, 1] = img[j + current_offset, i, 1]
                            new_img[j, i, 2] = img[j + current_offset, i, 2]
                    elif offset_where == 1:  # y
                        if (current_offset_abs + i) < width:
                            new_img[j, i, 0] = img[j, i + current_offset, 0]
                            new_img[j, i, 1] = img[j, i + current_offset, 1]
                            new_img[j, i, 2] = img[j, i + current_offset, 2]
                    elif offset_where == 2:  # x, y
                        if (current_offset_abs + j) < height:
                            if (current_offset_abs + i) < width:
                                new_img[j, i, 0] = img[j + current_offset, i + current_offset, 0]
                                new_img[j, i, 1] = img[j + current_offset, i + current_offset, 1]
                                new_img[j, i, 2] = img[j + current_offset, i + current_offset, 2]
    return new_img


def effect_horiz_shift(img):
    """
    Horizontal Shift effect: take entire set of rows and shift over,
    take entire selected row patch (y1 to y2) left;
    Increase offset each shift, but if overflow-> kick over to right side (width-current)
    :param img:
    :return new_img:
    """
    height = img.shape[0]  # j, patch_y
    width = img.shape[1]  # i, patch_x
    new_img = np.copy(img)
    patches = random.randrange(3, 7)
    low = 1.07
    med = 1.10
    high = 1.2
    divisor = low

    # Loop thru patches:
    for p in range(patches):
        # Get patch size, shift amt, and end of patch window
        patch_y = random.randrange(5, int(height // 1.2))
        patch_y_size = random.randrange(10, int(height // 4))
        shift = random.randrange(10, int(width // 5))
        patchend = patch_y + patch_y_size
        if patchend >= height:  # ensure within size limit
            patchend = height - 1

        # Randomize color channel to focus on:
        color_focus = random.randrange(0, 4)

        # Get divisor amount:
        divisor_selector = random.randrange(0, 3)
        if divisor_selector == 0:
            divisor = low
        elif divisor_selector == 1:
            divisor = med
        elif divisor_selector == 2:
            divisor = high

        # Loop thru patch window
        for i in range(width):
            for j in range(patch_y, patchend):
                # Get actual shift amount, and determine if overflow:
                shift_result = i - shift
                if shift_result < 0:
                    shift_result = width + shift_result
                if color_focus == 0:
                    new_img[j, shift_result, 0] = img[j, i, 0]
                    new_img[j, shift_result, 1] = int(img[j, i, 1] // divisor)
                    new_img[j, shift_result, 2] = int(img[j, i, 2] // divisor)
                elif color_focus == 1:
                    new_img[j, shift_result, 0] = int(img[j, i, 0] // divisor)
                    new_img[j, shift_result, 1] = img[j, i, 1]
                    new_img[j, shift_result, 2] = int(img[j, i, 2] // divisor)
                elif color_focus == 2:
                    new_img[j, shift_result, 0] = int(img[j, i, 0] // divisor)
                    new_img[j, shift_result, 1] = int(img[j, i, 1] // divisor)
                    new_img[j, shift_result, 2] = img[j, i, 2]
                else:
                    # ORIG w/o color band
                    new_img[j, shift_result, 0] = img[j, i, 0]
                    new_img[j, shift_result, 1] = img[j, i, 1]
                    new_img[j, shift_result, 2] = img[j, i, 2]
    return new_img
