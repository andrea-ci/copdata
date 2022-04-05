# -*- coding: utf-8 -*-
from os import path, listdir
import numpy as np
import pandas as pd
import cv2 as cv
from . import geoutils as gt
from . import utils

# Init the logger.
logger = utils.get_logger()

class SentinelProduct:
    """Loads and processes Sentinel products.

    Currently Sentinel-1 and Sentinel-2 products are supported.
    """

    def __init__(self, metadata, dir_root = '.'):

        self.metadata = metadata
        self.data_path = path.join(dir_root, metadata['dataFolder'])
        self.img_shape = self.check_data()
        self.img_coords = self.eval_img_coords()

    @property
    def images(self):

        res = {}

        mission = self.metadata['mission']

        path_images = self._get_path_images()

        if mission == 'sentinel-1':

            path_image_vv = path_images['vv']
            path_image_vh = path_images['vh']

            # Load Sentinel-1 images.
            image_vv = cv.imread(path_image_vv, cv.IMREAD_UNCHANGED)
            image_vh = cv.imread(path_image_vh, cv.IMREAD_UNCHANGED)

            res['vv'] = image_vv
            res['vh'] = image_vh

        else:
            raise NotImplementedError

        return res

    def check_data(self):
        """Checks access to image(s) and returns its/their size."""

        data_path = self.data_path
        mission = self.metadata['mission']

        # Load images.
        imgs = self.images

        if mission == 'sentinel-1':

            assert imgs['vv'].shape == imgs['vh'].shape
            n_rows, n_cols = imgs['vv'].shape

        return n_rows, n_cols

    def eval_img_coords(self):
        """Evaluates coordinates long/lat of image pixels."""

        poly_img = self.metadata['polygon']
        n_rows, n_cols = self.img_shape

        img_coords = gt.get_img_coords(poly_img, n_rows, n_cols)

        return img_coords

    def extract_aoi(self, poly_aoi):
        """Extracts a given AOI from the images."""

        direction = self.metadata['passDirection']
        mission = self.metadata['mission']

        path_images = self._get_path_images()
        n_rows, n_cols = self.img_shape

        # Gets geo-coordinates of pixels.
        img_coords = self.img_coords

        # Evaluates the bbox of AOI.
        longs = [pnt[0] for pnt in poly_aoi]
        lats = [pnt[1] for pnt in poly_aoi]
        lat_min = min(lats)
        lat_max = max(lats)
        long_min = min(longs)
        long_max = max(longs)
        bbox = (long_min, lat_min, long_max, lat_max)

        r_min, r_max, c_min, c_max = gt.find_bbox_inds(img_coords, bbox)
        
        # Load images.
        imgs = self.images

        if mission == 'sentinel-1':

            image_vv = imgs['vv']
            image_vh = imgs['vh']

            # Adjust the image according to the pass direction.
            if direction == 'descending':
                # Flip along East-West axis.
                image_vv = image_vv[:, ::-1]
                image_vh = image_vh[:, ::-1]
            else:
                # Flip along North-South axis.
                image_vv = image_vv[::-1, :]
                image_vh = image_vh[::-1, :]

            cropped_vv = image_vv[r_min : r_max + 1, c_min : c_max + 1]
            cropped_vh = image_vh[r_min : r_max + 1, c_min : c_max + 1]

        return cropped_vv, cropped_vh

    def _get_path_images(self):

        data_path = self.data_path
        mission = self.metadata['mission']

        path_images = {}

        if mission == 'sentinel-1':

            img_folder = path.join(data_path, 'measurement')

            for file in listdir(img_folder):

                filepath = path.join(img_folder, file)

                if '-vv-' in filepath:
                    path_images['vv'] = filepath
                elif '-vh-' in filepath:
                    path_images['vh'] = filepath

        elif mission == 'sentinel-2':
            raise NotImplementedError()

        else:
            raise ValueError('Mission not supported.')

        return path_images
