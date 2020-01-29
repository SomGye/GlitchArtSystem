import numpy as np
import random


def effect_soundwave(img, colorshift=25):
    """
    Create a variable 'sound-wave' by having random up, down, and right
    amounts.
    The wave simulates the switch b/w up and down modes.
    Each channel in the wave shifts the color
     within (-colorshift, colorshift) range.
    :param img:
    :param colorshift:
    :return:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    half_height = int(height // 2)
    new_img = np.copy(img)

    # Choose init wave params
    vert_selector = random.randrange(0, 2)
    up_amt = random.randrange(10, int(half_height // 1.2))
    down_amt = random.randrange(10, int(half_height // 1.2))
    right_amt = random.randrange(1, 11)
    for i in range(width):
        if i % right_amt == 0:  # switch dir
            # Reinit. wave params
            vert_selector = random.randrange(0, 2)
            up_amt = random.randrange(10, int(half_height // 1.2))
            down_amt = random.randrange(10, int(half_height // 1.2))
            right_amt = random.randrange(1, 11)
        if vert_selector == 0:  # go up
            for j in range(half_height, half_height - up_amt, -1):
                # Randomize color
                newb = img[j][i][0] + (random.randrange(-colorshift, colorshift))
                newg = img[j][i][1] + (random.randrange(-colorshift, colorshift))
                newr = img[j][i][2] + (random.randrange(-colorshift, colorshift))
                # Apply color
                new_img[j][i][0] = newb
                new_img[j][i][1] = newg
                new_img[j][i][2] = newr
        elif vert_selector == 1:  # go down
            for j in range(half_height, half_height + down_amt):
                # Randomize color
                newb = img[j][i][0] + (random.randrange(-colorshift, colorshift))
                newg = img[j][i][1] + (random.randrange(-colorshift, colorshift))
                newr = img[j][i][2] + (random.randrange(-colorshift, colorshift))
                # Apply color
                new_img[j][i][0] = newb
                new_img[j][i][1] = newg
                new_img[j][i][2] = newr

        # for j in range(height):
    return new_img


def effect_static(img):
    """
    Static effect: randomize 'pock' marks of random greyscale values;
    Loop 1: cover whole image with semi-uniform specks of random color
    Loop 2: choose random spots and fill those in too
    :param img:
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    max_color = 255

    # Loop 1
    for i in range(width):
        iend = int(width // 40)  # need to tweak
        jend = int(height // 40)
        if iend <= 3:
            iend = 6
        if jend <= 3:
            jend = 6
        spacingi = random.randrange(3, iend)  # was 60,60
        spacingj = random.randrange(3, jend)
        for j in range(height):
            if ((i % spacingi) == 0) and ((j % spacingj) == 0):
                # if (i % spacingi == 0): #solid vert. lines
                use_color = random.randrange(0, 4)
                if use_color == 0:
                    new_img[j, i, 0] = img[j, i, 0] // 3
                    new_img[j, i, 1] = img[j, i, 1] // 3
                    new_img[j, i, 2] = img[j, i, 2] // 3
                else:
                    spot_color = random.randrange(int(max_color // 7), int(max_color // 1.5))  # roughly 42->159
                    new_img[j, i, 0] = spot_color
                    new_img[j, i, 1] = spot_color
                    new_img[j, i, 2] = spot_color
    # Loop 2
    random_spots = random.randrange(90, 180)
    for r in range(random_spots):
        use_square = random.randrange(0, 3)  # 0=regular, 1-2=use square patch
        randj = random.randrange(5, height - 5)
        randi = random.randrange(5, width - 5)
        spot_color = random.randrange(int(max_color // 7), int(max_color // 1.5))  # roughly 42->159
        if use_square == 0:
            new_img[randj, randi, 0] = spot_color
            new_img[randj, randi, 1] = spot_color
            new_img[randj, randi, 2] = spot_color
        else:
            new_img[randj, randi, 0] = spot_color
            new_img[randj, randi, 1] = spot_color
            new_img[randj, randi, 2] = spot_color

            new_img[randj - 1, randi - 1, 0] = spot_color
            new_img[randj - 1, randi - 1, 1] = spot_color
            new_img[randj - 1, randi - 1, 2] = spot_color

            new_img[randj + 1, randi + 1, 0] = spot_color
            new_img[randj + 1, randi + 1, 1] = spot_color
            new_img[randj + 1, randi + 1, 2] = spot_color

            new_img[randj - 1, randi + 1, 0] = spot_color
            new_img[randj - 1, randi + 1, 1] = spot_color
            new_img[randj - 1, randi + 1, 2] = spot_color

            new_img[randj + 1, randi - 1, 0] = spot_color
            new_img[randj + 1, randi - 1, 1] = spot_color
            new_img[randj + 1, randi - 1, 2] = spot_color
    return new_img


def effect_scanlines(img, lines=5):
    """
    Scanlines effect: horizontal lines of solid grayscale static,
    with bursts of color
    :param img:
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    max_color = 255
    if lines <= 5:
        lines = random.randrange(5, 25)
    for l in range(lines):
        randomj = random.randrange(int(height // 12), int(height // 1.2))
        for i in range(width):
            spacingj = random.randrange(int(height // 12), int(height // 1.2))
            for j in range(height):
                if (j % spacingj) == 0:
                    use_color = random.randrange(0, 4)  # determine if g.s. or not
                    use_thick_line = random.randrange(0, 4)  # determine if thicker or not
                    if use_color == 0:
                        # Copy initial color:
                        colorvalb = img[randomj, i, 0]
                        colorvalg = img[randomj, i, 1]
                        colorvalr = img[randomj, i, 2]
                        # Determine dominant color and half others:
                        color_max = max([colorvalb, colorvalg, colorvalr])
                        if color_max == colorvalb:
                            new_img[randomj, i, 0] = colorvalb
                            new_img[randomj, i, 1] = colorvalg // 2
                            new_img[randomj, i, 2] = colorvalr // 2
                            if use_thick_line != 0:
                                new_img[randomj + 1, i, 0] = colorvalb
                                new_img[randomj + 1, i, 1] = colorvalg // 2
                                new_img[randomj + 1, i, 2] = colorvalr // 2
                        elif color_max == colorvalg:
                            new_img[randomj, i, 0] = colorvalb // 2
                            new_img[randomj, i, 1] = colorvalg
                            new_img[randomj, i, 2] = colorvalr // 2
                            if use_thick_line != 0:
                                new_img[randomj + 1, i, 0] = colorvalb // 2
                                new_img[randomj + 1, i, 1] = colorvalg
                                new_img[randomj + 1, i, 2] = colorvalr // 2
                        elif color_max == colorvalr:
                            new_img[randomj, i, 0] = colorvalb // 2
                            new_img[randomj, i, 1] = colorvalg // 2
                            new_img[randomj, i, 2] = colorvalr
                            if use_thick_line != 0:
                                new_img[randomj + 1, i, 0] = colorvalb // 2
                                new_img[randomj + 1, i, 1] = colorvalg // 2
                                new_img[randomj + 1, i, 2] = colorvalr
                    else:
                        spot_color = random.randrange(int(max_color // 6), int(max_color // 1.6))  # roughly 42->159
                        new_img[randomj, i, 0] = spot_color  # was j,i
                        new_img[randomj, i, 1] = spot_color
                        new_img[randomj, i, 2] = spot_color
                        if use_thick_line != 0:
                            new_img[randomj + 1, i, 0] = spot_color  # was j,i
                            new_img[randomj + 1, i, 1] = spot_color
                            new_img[randomj + 1, i, 2] = spot_color
    return new_img


def effect_copy_over(img, orig_img, option=0):
    """
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
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    # Choose action depending on option
    if option == 0:
        option = random.randrange(1, 11)  # 9,10 = nothing!
    if option == 1:  # top half
        for i in range(width):
            for j in range(height // 2):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 2:  # bottom half
        for i in range(width):
            for j in range(height // 2, height):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 3:  # left half
        for i in range(width // 2):
            for j in range(height):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 4:  # right half
        for i in range(width // 2, width):
            for j in range(height):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 5:  # top left
        for i in range(width // 2):
            for j in range(height // 2):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 6:  # top right
        for i in range(width // 2, width):
            for j in range(height // 2):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 7:  # bottom left
        for i in range(width // 2):
            for j in range(height // 2, height):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 8:  # bottom right
        for i in range(width // 2, width):
            for j in range(height // 2, height):
                new_img[j, i, 0] = orig_img[j, i, 0]
                new_img[j, i, 1] = orig_img[j, i, 1]
                new_img[j, i, 2] = orig_img[j, i, 2]
    return new_img


def effect_cross_hatch(img, random_spots=150):
    """
    Random hatch spots in variable X patterns.
    Similar to static.
    :param img:
    :param random_spots:
    :return new_img:
    """
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx
    new_img = np.copy(img)
    max_color = 255

    # Loop
    if random_spots == 150:
        random_spots = random.randrange(150, 301)
    for r in range(random_spots):
        randj = random.randrange(10, height - 10)
        randi = random.randrange(10, width - 10)
        spot_color = random.randrange(int(max_color // 8), int(max_color // 1.2))

        new_img[randj, randi, 0] = spot_color
        new_img[randj, randi, 1] = spot_color
        new_img[randj, randi, 2] = spot_color
        # Begin series of 8 dots for Cross pattern,
        # with each spot being optional for variation
        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj - 1, randi - 1, 0] = spot_color
            new_img[randj - 1, randi - 1, 1] = spot_color
            new_img[randj - 1, randi - 1, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj + 1, randi + 1, 0] = spot_color
            new_img[randj + 1, randi + 1, 1] = spot_color
            new_img[randj + 1, randi + 1, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj - 1, randi + 1, 0] = spot_color
            new_img[randj - 1, randi + 1, 1] = spot_color
            new_img[randj - 1, randi + 1, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj + 1, randi - 1, 0] = spot_color
            new_img[randj + 1, randi - 1, 1] = spot_color
            new_img[randj + 1, randi - 1, 2] = spot_color
        # --
        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj - 3, randi - 3, 0] = spot_color
            new_img[randj - 3, randi - 3, 1] = spot_color
            new_img[randj - 3, randi - 3, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj + 3, randi + 3, 0] = spot_color
            new_img[randj + 3, randi + 3, 1] = spot_color
            new_img[randj + 3, randi + 3, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj - 3, randi + 3, 0] = spot_color
            new_img[randj - 3, randi + 3, 1] = spot_color
            new_img[randj - 3, randi + 3, 2] = spot_color

        do_spot = random.randrange(0, 3)
        if do_spot != 0:
            new_img[randj + 3, randi - 3, 0] = spot_color
            new_img[randj + 3, randi - 3, 1] = spot_color
            new_img[randj + 3, randi - 3, 2] = spot_color
    return new_img


def effect_copy_over_color_distort(img, orig_img, option=0):
    """
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
    :return new_img:
    """
    new_img = np.copy(img)
    height = img.shape[0]  # j, patchy
    width = img.shape[1]  # i, patchx

    # Choose action depending on option
    if option == 0:
        option = random.randrange(1, 11)  # 9,10 = nothing!
    if option == 1:  # top half
        for i in range(width):
            # Copy initial color:
            colorvalb = img[0, i, 0]
            colorvalg = img[0, i, 1]
            colorvalr = img[0, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 2:  # bottom half
        for i in range(width):
            # Copy initial color:
            colorvalb = img[height // 2, i, 0]
            colorvalg = img[height // 2, i, 1]
            colorvalr = img[height // 2, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2, height):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 3:  # left half
        for i in range(width // 2):
            # Copy initial color:
            colorvalb = img[0, i, 0]
            colorvalg = img[0, i, 1]
            colorvalr = img[0, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 4:  # right half
        for i in range(width // 2, width):
            # Copy initial color:
            colorvalb = img[0, i, 0]
            colorvalg = img[0, i, 1]
            colorvalr = img[0, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 5:  # top left
        for i in range(width // 2):
            # Copy initial color:
            colorvalb = img[0, i, 0]
            colorvalg = img[0, i, 1]
            colorvalr = img[0, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 6:  # top right
        for i in range(width // 2, width):
            # Copy initial color:
            colorvalb = img[0, i, 0]
            colorvalg = img[0, i, 1]
            colorvalr = img[0, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 7:  # bottom left
        for i in range(width // 2):
            # Copy initial color:
            colorvalb = img[height // 2, i, 0]
            colorvalg = img[height // 2, i, 1]
            colorvalr = img[height // 2, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2, height):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    elif option == 8:  # bottom right
        for i in range(width // 2, width):
            # Copy initial color:
            colorvalb = img[height // 2, i, 0]
            colorvalg = img[height // 2, i, 1]
            colorvalr = img[height // 2, i, 2]
            # Determine dominant color and half others:
            color_max = max([colorvalb, colorvalg, colorvalr])
            for j in range(height // 2, height):
                if color_max == colorvalb:
                    new_img[j, i, 0] = orig_img[j, i, 0]
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2] // 2
                elif color_max == colorvalg:
                    new_img[j, i, 0] = int(orig_img[j, i, 0] // 1.3)
                    new_img[j, i, 1] = orig_img[j, i, 1]
                    new_img[j, i, 2] = int(orig_img[j, i, 2] // 1.3)
                else:
                    new_img[j, i, 0] = orig_img[j, i, 0] // 2
                    new_img[j, i, 1] = int(orig_img[j, i, 1] // 1.3)
                    new_img[j, i, 2] = orig_img[j, i, 2]
    return new_img
