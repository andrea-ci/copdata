# -*- coding: utf-8 -*-
"""This script perform a query to Copernicus, filters and downloads valid products
and extracts to their SAFE folders. """
from core import SatEngine

fn_geojson = 'geoareas/naples.geojson'
date_start = '20170101'
date_end = '20211231'

# Init the list of downloaded products.
engine = SatEngine('andrea32874763', 'yigXUR6M!eunDdv',
    dir_downloads = '/hdd2/sentinel-products/downloads',
    dir_products = '/hdd2/sentinel-products/data',
    dir_images = '/hdd2/sentinel-products/images')

engine.from_csv('napoli-2017-2021.csv')
engine.download(fn_geojson, date_start, date_end, policy = 'Contains',
    check_only = False, query = False)
#engine.to_csv('napoli-2017-2021.csv')
exit()

engine.extract_aoi(fn_geojson)
res = engine.detect_ships()
