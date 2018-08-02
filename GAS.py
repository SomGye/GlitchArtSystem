import random
import sys
# import os
import numpy as np
import cv2

import GAS_Effects as GAS

#   Author: Maxwell Crawford
#   CSC475 Final: Glitch Art Generator --> GlitchArtSystem
#   8-1-18
#   2018 vers. 3a - refactor and add command-line support

## BEGIN CALLS/SETUP
if __name__ == "__main__":
    """Main program entry point for GAS.
    Raises:
        TypeError -- When no suitable image type found.    
    Returns:
        None
    """

    # Grab any command-line args...
    script_args_len = len(sys.argv)
    # script_args = str(sys.argv)
    use_arg = False # default

    # Setup Random Call List for effects order
    supported_effects = 15 #currently supported effects...
    layers_num = random.randrange(5, 26)
    current_effects = []
    print("* Generating Random Order of Effects...")
    for e in range(layers_num):
        # Generate effect layer #:
        effect = random.randrange(1, supported_effects+1)
        current_effects.append(effect) #add resulting effect to list

    #Generate random list from chosen effects,
    # ensuring two conditions:
    randlist = []
    for e in range(len(current_effects)):
        choice = random.choice(current_effects)
        #1) Prevent CopyOver (10) from being 1st effect!
        if (e == 0):
            while True:
                if (choice == 10):
                    choice = random.randrange(1, supported_effects + 1)
                    continue
                else:
                    break
        else:
            #2) If effect is same as last one, try again:
            while True:
                if (choice == randlist[-1]):
                    choice = random.randrange(1, supported_effects + 1)
                    continue
                else:
                    break
        randlist.append(choice)  # non-unique choices after conditions

    # Check if Command-Line args were given:
    if script_args_len > 1:
        use_arg = True

    # Check user input with drag-n-drop:
    dragged_name = input("Drag your file here -->\t")
    
    # Load Images
    # 1) Check sys arg for user-defined image
    # 2) Check test.png (in images folder or not)
    # 3) Check test.jpg (in images folder or not)
    # 4) Else, raise error.
    img1 = None
    print("* Loading initial image...")
    if use_arg:
        # if os.path.exists()
        try:
            img1 = cv2.imread(str(sys.argv[1]))
        except:
            use_arg = False # break
    else:
        img1 = cv2.imread(dragged_name)
        img1_g = cv2.imread(dragged_name, 0)
        if img1 is None:
            img1 = cv2.imread("images/test.png")
            img1_g = cv2.imread("images/test.png", 0)
        if img1 is None:
            img1 = cv2.imread("images/test.jpg")
            img1_g = cv2.imread("images/test.jpg", 0)
        if img1 is None:
            img1 = cv2.imread("test.png")
            img1_g = cv2.imread("test.png", 0)
        if img1 is None:
            img1 = cv2.imread("test.jpg")
            img1_g = cv2.imread("test.jpg", 0)
        if img1 is None:
            raise TypeError('No valid image found or specified.\n')

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
        if effect == 1: #call 1st effect
            print("-- Effect: Random Pixel Shift")
            newimg = GAS.effectRandomPixelShift(img)
        elif effect == 2:
            print("-- Effect: Color Smear")
            newimg = GAS.effectColorSmear(img)
        elif effect == 3:
            print("-- Effect: Color Scratch")
            scratchdir = random.randrange(0, 2)
            newimg = GAS.effectColorScratch(img, 0, 0, 4, scratchdir)
        elif effect == 4:
            print("-- Effect: SoundWave")
            newimg = GAS.effectSoundWave(img)
        elif effect == 5:
            print("-- Effect: Static")
            newimg = GAS.effectStatic(img)
        elif effect == 6:
            print("-- Effect: Scanlines")
            newimg = GAS.effectScanlines(img)
        elif effect == 7:
            print("-- Effect: Horiz. Shift")
            newimg = GAS.effectHorizShift(img)
        elif effect == 8:
            print("-- Effect: Color Compression Bands")
            newimg = GAS.effectColorCompression(img)
        elif effect == 9:
            print("-- Effect: Harris Color Shift")
            newimg = GAS.effectHarrisEdgeColorShift(img, newimg_g)
        elif effect == 10:
            print("-- Effect: Copy Over (Original Vers.)")
            newimg = GAS.copyOver(img, img1)
        elif effect == 11:
            print("-- Effect: Convolution Edge Lines")
            newimg = GAS.effectConvolutionEdgeLines(img)
        elif effect == 12:
            print("-- Effect: Convolution Edge Dilation")
            newimg = GAS.effectConvolutionEdgeDilation(img)
        elif effect == 13:
            print("-- Effect: Convolution Dynamics")
            newimg = GAS.effectConvolutionDynamic(img)
        elif effect == 14:
            print("-- Effect: Cross Hatch")
            newimg = GAS.effectCrossHatch(img)
        elif effect == 15:
            print("-- Effect: Copy Over (Color Distort)")
            newimg = GAS.copyOverColorDistort(img, img1)
        return newimg

    # Seed the random library
    random.seed()

    # Store and generate random 'hash' to append to resulting filename:
    file_hash = ''
    hash_list = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'a', 'b', 'c', 'd', 'e', 'f',
        'A', 'B', 'C', 'D', 'E', 'F'
        ]
    hash_length = 10
    for h in range(hash_length):
        current_hash_num = random.randrange(0, len(hash_list))
        current_hash_item = hash_list[current_hash_num]
        file_hash += current_hash_item
    
    # Compose resulting filename:
    result_file_path = "results/resulttest_"
    result_file_path += file_hash 
    result_file_path += ".png"
    ## END SETUP

    # Display Results
    large = 1600
    result = np.copy(img1)
    for e in range(len(randlist)):
        print("\n* Performing Effect Layer #" + str(e+1) + " / " + str(layers_num))
        result = effectCaller2(result, randlist[e])
    # cv2.imwrite("results/resulttest.png", result, [cv2.IMWRITE_PNG_COMPRESSION,0]) #note the PNG, lowest compression!
    cv2.imwrite(result_file_path, result, \
        [cv2.IMWRITE_PNG_COMPRESSION, 0]) #note the PNG, lowest compression!
    cv2.imshow("Original Image", img1)
    if img1.shape[0] > large: #If image is 'large', then resize for convenience
        result = cv2.resize(result, (int(0.5*result.shape[1]), int(0.5*result.shape[0])), interpolation=cv2.INTER_AREA)
    cv2.imshow("Effect Caller2 Result", result)
    cv2.waitKey(0)