import cv2
import numpy as np
import random

import GAS_Effects as GAS

#Author: Maxwell Crawford
#CSC475 Final: Glitch Art Generator
#5-3-18
#2018 vers. 2

## BEGIN CALLS/SETUP
if __name__ == "__main__":
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
            newimg = GAS.effectRandomPixelShift(img)
        elif (effect == 2):
            print("-- Effect: Color Smear")
            newimg = GAS.effectColorSmear(img)
        elif (effect == 3):
            print("-- Effect: Color Scratch")
            scratchdir = random.randrange(0, 2)
            newimg = GAS.effectColorScratch(img, 0, 0, 4, scratchdir)
        elif (effect == 4):
            print("-- Effect: SoundWave")
            newimg = GAS.effectSoundWave(img)
        elif (effect == 5):
            print("-- Effect: Static")
            newimg = GAS.effectStatic(img)
        elif (effect == 6):
            print("-- Effect: Scanlines")
            newimg = GAS.effectScanlines(img)
        elif (effect == 7):
            print("-- Effect: Horiz. Shift")
            newimg = GAS.effectHorizShift(img)
        elif (effect == 8):
            print("-- Effect: Color Compression Bands")
            newimg = GAS.effectColorCompression(img)
        elif (effect == 9):
            print("-- Effect: Harris Color Shift")
            newimg = GAS.effectHarrisEdgeColorShift(img, newimg_g)
        elif (effect == 10):
            print("-- Effect: Copy Over (Original Vers.)")
            newimg = GAS.copyOver(img, img1)
        elif (effect == 11):
            print("-- Effect: Convolution Edge Lines")
            newimg = GAS.effectConvolutionEdgeLines(img)
        elif (effect == 12):
            print("-- Effect: Convolution Edge Dilation")
            newimg = GAS.effectConvolutionEdgeDilation(img)
        elif (effect == 13):
            print("-- Effect: Convolution Dynamics")
            newimg = GAS.effectConvolutionDynamic(img)
        elif (effect == 14):
            print("-- Effect: Cross Hatch")
            newimg = GAS.effectCrossHatch(img)
        elif (effect == 15):
            print("-- Effect: Copy Over (Color Distort)")
            newimg = GAS.copyOverColorDistort(img, img1)
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