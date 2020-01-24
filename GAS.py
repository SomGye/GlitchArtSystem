import random
from os import path
import sys
from datetime import date
import numpy as np
import cv2

from filters import color_filters, location_filters, complex_filters

#   Author: Maxwell Crawford
#   GlitchArtSystem
#   2020-01-22
#   2020 v2 - Refactor to PEP8 std., modular design

DEBUG = True

if __name__ == "__main__":
    """Main program entry point for GAS.
    Raises:
        TypeError -- When no suitable image type found.    
    Returns:
        None
    """
    print("*~~~ Welcome to GlitchArtSystem! ~~~*\n")
    # Check Debug Flag - Disable Traceback for Release
    if not DEBUG:
        sys.tracebacklimit = 0
    else:
        print('### DEBUG MODE ###')

    # Grab any command-line args...
    script_args_len = len(sys.argv)
    use_arg = False  # default

    # Setup Random Call List for effects order
    supported_effects = 15  # currently supported effects...
    layers_num = random.randrange(5, 26)
    current_effects = []
    print("* Generating Random Order of Effects...")
    for e in range(layers_num):
        # Generate effect layer #:
        current_effect = random.randrange(1, supported_effects + 1)
        current_effects.append(current_effect)  # add resulting effect to list

    # Generate random list from chosen effects,
    # ensuring two conditions:
    randlist = []
    for e in range(len(current_effects)):
        choice = random.choice(current_effects)
        # 1) Prevent CopyOver (10) from being 1st effect!
        if e == 0:
            while True:
                if choice == 10:
                    choice = random.randrange(1, supported_effects + 1)
                    continue
                else:
                    break
        else:
            # 2) If effect is same as last one, try again:
            while True:
                if choice == randlist[-1]:
                    choice = random.randrange(1, supported_effects + 1)
                    continue
                else:
                    break
        randlist.append(choice)  # non-unique choices after conditions

    # Check if Command-Line args were given:
    dragged_path = None
    if script_args_len > 1:
        use_arg = True
    else:
        # Check user input with drag-n-drop:
        dragged_path = input("Drag your file here -->\t")

    # Load Images
    # 1) Check sys arg for user-defined image
    # 2) Check dragged path
    # 3) Check test path
    # 4) Else, raise error.
    file_path = ''
    test_path = 'test.jpg'
    src_img = None
    src_img_grayscale = None
    file_valid = False
    print("* Loading initial image...")
    if use_arg:
        file_path = str(sys.argv[1])
        src_img = cv2.imread(file_path)
        src_img_grayscale = cv2.imread(file_path, 0)

    # Check that path exists and is a file:
    if path.exists(file_path) and path.isfile(file_path):
        file_valid = True
    if not file_valid:
        # Check validity of dragged_path
        if path.exists(dragged_path) and path.isfile(dragged_path):
            # Try to use drag input file
            src_img = cv2.imread(dragged_path)
            src_img_grayscale = cv2.imread(dragged_path, 0)  # no color
        elif path.exists(test_path) and path.isfile(test_path):
            # Fallback options
            src_img = cv2.imread(test_path)
            src_img_grayscale = cv2.imread(test_path, 0)
        else:
            # All Else Failed!
            raise TypeError('No valid image found or specified.\n')


    def effect_caller(img, effect):
        """
        Calls the effect functions and displays basic info.
        This takes a copy of the existing layered image and
        returns an image with a new effect layer applied.
        :param img: source image (a NumPy array copy of a CV2 read file)
        :param effect: specific effect filter function num.
        :return new_img:
        """
        try:
            new_img = np.copy(img)
            new_img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except TypeError:
            raise TypeError('\nInvalid Image Matrix Detected - Try another file.')
        if effect == 1:  # call 1st effect
            print("-- Effect: Random Pixel Shift")
            new_img = location_filters.effect_random_pixel_shift(img)
        elif effect == 2:
            print("-- Effect: Color Smear")
            new_img = color_filters.effect_color_smear(img)
        elif effect == 3:
            print("-- Effect: Color Scratch")
            scratch_dir = random.randrange(0, 2)
            new_img = color_filters.effect_color_scratch(img, 0, 0, 4, scratch_dir)
        elif effect == 4:
            print("-- Effect: SoundWave")
            new_img = complex_filters.effect_soundwave(img)
        elif effect == 5:
            print("-- Effect: Static")
            new_img = complex_filters.effect_static(img)
        elif effect == 6:
            print("-- Effect: Scanlines")
            new_img = complex_filters.effect_scanlines(img)
        elif effect == 7:
            print("-- Effect: Horiz. Shift")
            new_img = location_filters.effect_horiz_shift(img)
        elif effect == 8:
            print("-- Effect: Color Compression Bands")
            new_img = color_filters.effect_color_compression(img)
        elif effect == 9:
            print("-- Effect: Harris Color Shift")
            new_img = color_filters.effect_harris_edge_color_shift(img, new_img_g)
        elif effect == 10:
            print("-- Effect: Copy Over (Original Vers.)")
            new_img = complex_filters.effect_copy_over(img, src_img)
        elif effect == 11:
            print("-- Effect: Convolution Edge Lines")
            new_img = color_filters.effect_convolution_edge_lines(img)
        elif effect == 12:
            print("-- Effect: Convolution Edge Dilation")
            new_img = color_filters.effect_convolution_edge_dilation(img)
        elif effect == 13:
            print("-- Effect: Convolution Dynamics")
            new_img = color_filters.effect_convolution_dynamic(img)
        elif effect == 14:
            print("-- Effect: Cross Hatch")
            new_img = complex_filters.effect_cross_hatch(img)
        elif effect == 15:
            print("-- Effect: Copy Over (Color Distort)")
            new_img = complex_filters.effect_copy_over_color_distort(img, src_img)
        return new_img


    # Seed the random library
    random.seed()

    # Store and generate random 'hash' to append to resulting filename:
    file_hash = ''
    hash_list = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        'a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'j', 'k',
        'A', 'B', 'C', 'D', 'E', 'F', 'H', 'I', 'J', 'K'
    ]
    hash_length = 12
    for h in range(hash_length):
        current_hash_num = random.randrange(0, len(hash_list))
        current_hash_item = hash_list[current_hash_num]
        file_hash += current_hash_item

    # Compose resulting filename from hash plus ISO date:
    today = str(date.today())
    result_file_path = "results/resulttest_"
    result_file_path += file_hash
    result_file_path += '_' + today
    result_file_path += ".png"

    # Display Results
    large = 1600
    result = np.copy(src_img)
    for e in range(len(randlist)):
        print("\n* Performing Effect Layer #" + str(e + 1) + " / " + str(layers_num))
        result = effect_caller(result, randlist[e])
    cv2.imwrite(result_file_path, result,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])  # note the PNG, lowest compression!
    cv2.imshow("Original Image", src_img)

    # Scale preview output image down for convenience
    if src_img.shape[0] > large:
        result = cv2.resize(result, (int(0.5 * result.shape[1]), int(0.5 * result.shape[0])),
                            interpolation=cv2.INTER_AREA)
    cv2.imshow("Glitch Art Result", result)
    cv2.waitKey(0)
