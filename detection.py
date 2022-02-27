# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

@jit(nopython = True)
def cfar_detector(image, size_outer = 100, size_inner = 40, thr = 4):
    """Implements a Gaussian CFAR detector for amplitude images."""

    n_rows, n_cols = image.shape
    image_out = np.zeros((n_rows, n_cols), dtype = 'uint8')

    luside = int((size_outer - 1) / 2)
    rdside = size_outer - luside - 1
    hside = int((size_outer - size_inner) / 2)

    # Pre-allocated indices for background pixels.
    idx_rows = np.array(list(range(hside)) + list(range(hside + size_inner, size_outer)))
    idx_cols = np.array(list(range(hside)) + list(range(hside + size_inner, size_outer)))

    # Slide a window across the image.
    for ii in range(luside, n_rows - rdside):
        for jj in range(luside, n_cols - rdside):

            # Get the current window of pixels.
            full_window = image[ii - luside : ii + rdside + 1, jj - luside : jj + rdside + 1]

            # Extract background excluding the guard window.
            bg = np.hstack((full_window[idx_rows, :].ravel(),
                full_window[hside : hside + size_inner, idx_cols].ravel()))

            # Background mean value and std.
            bg_mean = np.mean(bg)
            bg_std = np.std(bg)

            # Apply gaussian threshold for pixel under test.
            if image[ii, jj] >= bg_mean + thr * bg_std:
                image_out[ii, jj] = 1

    return image_out

def label_image_with_boxes(filename, image, boxes):
    """Draws an image with boxes of detected objects."""

    # plot the image
    plt.figure()
    plt.imshow(image)

    # get the context for drawing boxes
    ax = plt.gca()

    # plot each box
    for box in boxes:

        # get coordinates
        x_min, y_min, x_max, y_max = box

        # calculate width and height of the box
        width, height = x_max - x_min, y_max - y_min

        # create the shape
        rect = Rectangle((x_min, y_min), width, height, fill = False, color = 'red')

        # draw the box
        ax.add_patch(rect)

    # Save the figure
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, dpi = 300)
    plt.close()

def contours_to_boxes(contours):
    """
    Converts contours into boxes.
    A box is a tuple given by x_min, x_max, ymin, ymax.
    """

    boxes = []
    for cont in contours:

        x_min = None
        x_max = None
        y_min = None
        y_max = None

        for pnt in cont:

            x, y = pnt[0]

            if x_min is None:
                x_min = x
            else:
                x_min = min(x_min, x)

            if x_max is None:
                x_max = x
            else:
                x_max = max(x_max, x)

            if y_min is None:
                y_min = y
            else:
                y_min = min(y_min, y)

            if y_max is None:
                y_max = y
            else:
                y_max = max(y_max, y)

        boxes.append((x_min, y_min, x_max, y_max))

    return boxes

def is_overlapping(box1, box2):
    """Checks if box1 is overlapping with box2."""

    res = True

    # Unpack coordinates of boxes.
    x_min_1, y_min_1, x_max_1, y_max_1 = box1
    x_min_2, y_min_2, x_max_2, y_max_2 = box2

    if x_max_1 < x_min_2 or x_max_2 < x_min_1:
        res = False

    elif y_max_1 < y_min_2 or y_max_2 < y_min_1:
        res = False

    return res

def merge_boxes(box1, box2):
    """Merge two overlapping boxes and returns the resulting box."""

    # Unpack coordinates of boxes.
    x_min_1, y_min_1, x_max_1, y_max_1 = box1
    x_min_2, y_min_2, x_max_2, y_max_2 = box2

    x_min = min(x_min_1, x_min_2)
    x_max = max(x_max_1, x_max_2)
    y_min = min(y_min_1, y_min_2)
    y_max = max(y_max_1, y_max_2)

    return (x_min, y_min, x_max, y_max)

def merge_overlapping_boxes(boxes):

    merged_boxes = []

    n_boxes = len(boxes)

    for ii in range(n_boxes):

        is_merged = False

        # Checking for ii-th box.
        for jj in range(ii + 1, n_boxes):

            if is_overlapping(boxes[ii], boxes[jj]):

                # Merge and replace the second box.
                merged = merge_boxes(boxes[ii], boxes[jj])
                boxes[jj] = merged

                is_merged = True

                # When ii-th has been merged, skip to next box to check.
                break

        if not is_merged:
            # If ii-th box is not overlapping with any other box, add it.
            merged_boxes.append(boxes[ii])

    return merged_boxes
