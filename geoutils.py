# -*- coding: utf-8 -*-
import numpy as np
from numba import jit
from turfpy import measurement
from geojson import Point, Feature, FeatureCollection, Polygon

EARTH_RADIUS = 6371000

def parse_polygon(polygon_str):
    """Parses the polygon string obtained by Copernicus product metadata.

    Returns the list of 4 (long, lat) coordinates of polygon verteces.
    """

    if not polygon_str.startswith('POLYGON'):
        raise ValueError('Invalid POLYGON string.')

    poly = []

    text = polygon_str[9:-2]
    list_coords_str = text.split(',')

    # Discard che last point as it's equal to the first one.
    for coords in list_coords_str[:-1]:
        long, lat = coords.split(' ')
        poly.append((float(long), float(lat)))

    return poly

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

def gen_pixel_coords(poly, n_rows, n_cols, dN_col, dE_col, dN_row, dE_row):

    img_coords = np.empty((n_rows, n_cols, 2))

    # Sort points by latitude.
    sorted_by_lat = sorted(poly, key = lambda x : x[1])

    # Take bottom points and pick the one with min longitude / bottom-left.
    upper_left = sorted(sorted_by_lat[2:], key = lambda x : x[0])[0]

    img_coords[0, 0] = upper_left

    img_coords = fill_pixel_coords(img_coords, n_rows, n_cols, dN_col, dE_col,
        dN_row, dE_row)

    return img_coords

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
