# -*- coding: utf-8 -*-
import logging
from logging import StreamHandler
import numpy as np

def get_logger():

    log_level = 'INFO'

    logger = logging.getLogger('s1-processing')

    file_log_formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(message)s',
        datefmt = '%Y-%m-%d %H:%M:%S')

    file_log_handler = StreamHandler()
    file_log_handler.setFormatter(file_log_formatter)

    logger.addHandler(file_log_handler)
    logger.setLevel(log_level)

    return logger

def create_image_rgb(image_vv, image_vh):
    """
    Creates a RGB image with images in VV and VH polarization.

    Red: VV
    Green: VH
    Blue: |VV| / |VH|
    """

    n_rows, n_cols = image_vv.shape
    image_rgb = np.zeros((n_rows, n_cols, 3), np.uint8)

    blue = np.divide(image_vv, image_vh, out = np.zeros_like(image_vv),
        where = image_vh != 0)

    # Compose the RGB image.
    image_rgb[:, :, 0] = image_vv
    image_rgb[:, :, 1] = image_vh
    image_rgb[:, :, 2] = blue

    return image_rgb
