# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
from turfpy import measurement
from geojson import Point, Feature, FeatureCollection, Polygon
from . import utils

# Init the logger.
logger = utils.get_logger()

EARTH_RADIUS = 6371000
PIXEL_SPACING = 10

def get_img_coords(img_poly, n_rows, n_cols):
    """Gets long/lat coordinates of pixels given polygon and size of the image."""

    # Sort points by latitude and identify some corners.
    sorted_by_lat = sorted(img_poly, key = lambda x : x[1])
    corner_se = sorted(sorted_by_lat[:2], key = lambda x : x[0])[0]
    corner_sw = sorted(sorted_by_lat[:2], key = lambda x : x[0])[1]
    corner_ne = sorted(sorted_by_lat[2:], key = lambda x : x[0])[0]
    corner_nw = sorted(sorted_by_lat[2:], key = lambda x : x[0])[1]

    # Calculate the angle between swath and N-S axis.
    theta1 = get_bearing(corner_sw, corner_nw)
    theta2 = get_bearing(corner_se, corner_ne)
    theta = np.mean([theta1, theta2])
    logger.info(f'Swath bearing is {theta}.')

    # Angle is negated because we must perform the opposite transformation
    # between coordinate systems, i.e. from swath-based to NS-EW.
    dE_col, dN_col = rotate_spacing(PIXEL_SPACING, 0, -theta)
    dE_row, dN_row = rotate_spacing(0, -PIXEL_SPACING, -theta)

    logger.info(f'Projection of image columns to East and North: {dE_col}m, {dN_col}m')
    logger.info(f'Projection of image rows to East and North: {dE_row}m, {dN_row}m')

    img_coords = np.empty((n_rows, n_cols, 2))
    img_coords[0, 0] = corner_ne

    img_coords = fill_pixel_coords(img_coords, n_rows, n_cols, dN_col, dE_col,
        dN_row, dE_row)

    return img_coords

def get_bearing(p1, p2):
    """Returns the bearing of swath. Bearing is the angle in degrees measured
    clockwise from north. """

    start = Feature(geometry = Point(p1))
    end = Feature(geometry = Point(p2))

    theta = measurement.bearing(start, end)

    return theta

def rotate_spacing(spacing_x, spacing_y, theta):
    """Returns the linear distance of a pixel along latitude and longitude.
    Theta is in degrees.
    """

    dx = spacing_x * np.cos(theta * np.pi / 180) - spacing_y * np.sin(theta * np.pi / 180)
    dy = spacing_x * np.sin(theta * np.pi / 180) + spacing_y * np.cos(theta * np.pi / 180)

    return dx, dy

@jit(nopython = True)
def add_meters_to_longlat(long_lat, dy, dx):

    long = long_lat[0]
    lat = long_lat[1]

    new_lat  = lat + (dy / EARTH_RADIUS) * (180 / np.pi)
    new_long = long + (dx / EARTH_RADIUS) * (180 / np.pi) / np.cos(lat * np.pi / 180)

    return new_long, new_lat

@jit(nopython = True)
def fill_pixel_coords(mat_coords, n_rows, n_cols, dN_col, dE_col, dN_row, dE_row):

    for ii in range(n_rows):

        if ii > 0:
            mat_coords[ii, 0] = add_meters_to_longlat(mat_coords[ii - 1, 0], dN_row, dE_row)

        for jj in range(1, n_cols):
            mat_coords[ii, jj] = add_meters_to_longlat(mat_coords[ii, jj - 1], dN_col, dE_col)

    return mat_coords

@jit(nopython = True)
def find_bbox_inds(img_coords, bbox):

    long_min, lat_min, long_max, lat_max = bbox

    n_rows, n_cols = img_coords.shape[:2]

    r_min = n_rows - 1
    r_max = 0
    c_min = n_cols - 1
    c_max = 0

    for ii in range(n_rows):
        for jj in range(n_cols):

            long, lat = img_coords[ii, jj, :]
            if long >= long_min and long <= long_max and lat >= lat_min and lat <= lat_max:

                r_min = min(r_min, ii)
                r_max = max(r_max, ii)
                c_min = min(c_min, jj)
                c_max = max(c_max, jj)

    return r_min, r_max, c_min, c_max
