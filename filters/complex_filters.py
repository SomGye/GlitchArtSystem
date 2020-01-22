import numpy as np
import random


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

    # Choose init wave params
    updown = random.randrange(0, 2)
    upamt = random.randrange(10, int(halfheight // 1.2))
    downamt = random.randrange(10, int(halfheight // 1.2))
    rightamt = random.randrange(1, 11)
    for i in range(width):
        if (i % rightamt == 0):  # switch dir
            # Reinit. wave params
            updown = random.randrange(0, 2)
            upamt = random.randrange(10, int(halfheight // 1.2))
            downamt = random.randrange(10, int(halfheight // 1.2))
            rightamt = random.randrange(1, 11)
        if (updown == 0):  # go up
            for j in range(halfheight, halfheight - upamt, -1):
                # Randomize color
                newb = img[j][i][0] + (random.randrange(-colorshift, colorshift))
                newg = img[j][i][1] + (random.randrange(-colorshift, colorshift))
                newr = img[j][i][2] + (random.randrange(-colorshift, colorshift))
                # Apply color
                newimg[j][i][0] = newb
                newimg[j][i][1] = newg
                newimg[j][i][2] = newr
        elif (updown == 1):  # go down
            for j in range(halfheight, halfheight + downamt):
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

    # Loop 1
    for i in range(width):
        iend = int(width // 40)  # need to tweak
        jend = int(height // 40)
        if (iend <= 3):
            iend = 6
        if (jend <= 3):
            jend = 6
        spacingi = random.randrange(3, iend)  # was 60,60
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
                    spotcolor = random.randrange(int(maxcolor // 7), int(maxcolor // 1.5))  # roughly 42->159
                    newimg[j, i, 0] = spotcolor
                    newimg[j, i, 1] = spotcolor
                    newimg[j, i, 2] = spotcolor
    # Loop 2
    randomspots = random.randrange(90, 180)
    for r in range(randomspots):
        useSquare = random.randrange(0, 3)  # 0=regular, 1-2=use square patch
        randj = random.randrange(5, height - 5)
        randi = random.randrange(5, width - 5)
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
                    useColor = random.randrange(0, 4)  # determine if g.s. or not
                    useThickLine = random.randrange(0, 4)  # determine if thicker or not
                    if (useColor == 0):
                        # Copy initial color:
                        colorvalb = img[randomj, i, 0]
                        colorvalg = img[randomj, i, 1]
                        colorvalr = img[randomj, i, 2]
                        # Determine dominant color and half others:
                        colormax = max([colorvalb, colorvalg, colorvalr])
                        if (colormax == colorvalb):
                            newimg[randomj, i, 0] = colorvalb
                            newimg[randomj, i, 1] = colorvalg // 2
                            newimg[randomj, i, 2] = colorvalr // 2
                            if (useThickLine != 0):
                                newimg[randomj + 1, i, 0] = colorvalb
                                newimg[randomj + 1, i, 1] = colorvalg // 2
                                newimg[randomj + 1, i, 2] = colorvalr // 2
                        elif (colormax == colorvalg):
                            newimg[randomj, i, 0] = colorvalb // 2
                            newimg[randomj, i, 1] = colorvalg
                            newimg[randomj, i, 2] = colorvalr // 2
                            if (useThickLine != 0):
                                newimg[randomj + 1, i, 0] = colorvalb // 2
                                newimg[randomj + 1, i, 1] = colorvalg
                                newimg[randomj + 1, i, 2] = colorvalr // 2
                        elif (colormax == colorvalr):
                            newimg[randomj, i, 0] = colorvalb // 2
                            newimg[randomj, i, 1] = colorvalg // 2
                            newimg[randomj, i, 2] = colorvalr
                            if (useThickLine != 0):
                                newimg[randomj + 1, i, 0] = colorvalb // 2
                                newimg[randomj + 1, i, 1] = colorvalg // 2
                                newimg[randomj + 1, i, 2] = colorvalr
                    else:
                        spotcolor = random.randrange(int(maxcolor // 6), int(maxcolor // 1.6))  # roughly 42->159
                        newimg[randomj, i, 0] = spotcolor  # was j,i
                        newimg[randomj, i, 1] = spotcolor
                        newimg[randomj, i, 2] = spotcolor
                        if (useThickLine != 0):
                            newimg[randomj + 1, i, 0] = spotcolor  # was j,i
                            newimg[randomj + 1, i, 1] = spotcolor
                            newimg[randomj + 1, i, 2] = spotcolor
    return newimg


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
        doSpot = random.randrange(0, 3)
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
        # --
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
