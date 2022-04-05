# -*- coding: utf-8 -*-
import numpy as np
#from scipy.stats import gamma as rv
from scipy.stats import norm as rv

def map_classification(img, err_tol = 0.01, n_classes = 3, scale = 5,
    max_iters = 10):
    """Coarse MAP classification of pixels."""

    n_rows, n_cols = img.shape[:2]
    mask = np.empty((n_rows, n_cols), dtype = 'uint8')

    img_min = np.min(img)
    img_max = np.max(img)

    # Intensity range of each class.
    class_intval = (img_max - img_min) / n_classes

    # Init PDF parameter arrays.
    #alpha_values = np.zeros((n_classes, max_iters))
    loc_values = np.zeros((n_classes, max_iters))
    scale_values = np.zeros((n_classes, max_iters))

    for ii in range(n_classes):

        x_start = img_min + ii * class_intval
        x_end = x_start + class_intval

        #alpha_values[ii, 0] = 5
        loc_values[ii, 0] = np.mean([x_start, x_end])
        # Exponential?
        scale_values[ii, 0] = scale

    # Prior probabilities for classes.
    probs_prior = np.ones(n_classes) / n_classes

    # Posterior probabilities for each pixel to belong to each class.
    probs_post = np.zeros((n_classes, n_rows, n_cols))

    n_iters = 0

    while True:

        print(f'Starting iteration no. {n_iters}.')

        # Estimate probabilities for all pixels to belong to each class.
        for ii in range(n_classes):

            #alpha = alpha_values[ii, n_iters]
            loc = loc_values[ii, n_iters]
            scale = scale_values[ii, n_iters]

            print(f'Parameters estimated for class {ii}: loc={loc}, scale={scale}.')
            probs_post[ii, :, :] = rv.pdf(img, loc = loc, scale = scale) * probs_prior[ii]

        for ii in range(n_classes):
            probs_post[ii, :, :] = np.divide(probs_post[ii, :, :],
                np.sum(probs_post, axis = 0))

        mask = np.argmax(probs_post, axis = 0)

        # Estimation of PDF parameters.
        for ii in range(n_classes):

            class_pixels = img[mask == ii]
            loc, scale = rv.fit(class_pixels)

            #alpha_values[ii, n_iters + 1] = alpha
            loc_values[ii, n_iters + 1] = loc
            scale_values[ii, n_iters + 1] = scale

        # Check exit conditions.
        #delta_alpha = np.max(np.abs(alpha_values[:, n_iters + 1] - alpha_values[:, n_iters]))
        delta_loc = np.max(np.abs(loc_values[:, n_iters + 1] - loc_values[:, n_iters]))
        delta_scale = np.max(np.abs(scale_values[:, n_iters + 1] - scale_values[:, n_iters]))

        #print(f'Max alpha difference: {delta_alpha}.')
        print(f'Max loc difference: {delta_loc}.')
        print(f'Max scale difference: {delta_scale}.')

        if n_iters > 0 and delta_loc < err_tol and delta_scale < err_tol:
            print('Convergence criteria are met, exiting.')
            break
        elif n_iters == max_iters:
            print('Reached max number of iterations, exiting.')
            break

        n_iters += 1

    return mask
