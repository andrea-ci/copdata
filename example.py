# -*- coding: utf-8 -*-
"""This script perform a query to Copernicus, filters and downloads valid products
and extracts to their SAFE folders. """
from os import path
import zipfile
import geoutils as gt
from products import ProductStore
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from config import PROD_LIST, DIR_DOWNLOADS, DIR_PRODUCTS, DIR_GEO
from core import ProductStore, SatEngine

fn_geojson = 'geoareas/naples.geojson'
date_start = '20220210'
date_end = '20220224'

# Init the list of downloaded products.
engine = SatEngine('andrea32874763', 'yigXUR6M!eunDdv')

#engine.download(fn_geojson, date_start, date_end, policy = 'Contains')
#engine.extract_aoi(fn_geojson)

engine.from_csv('prod_metadata.csv')
res = engine.detect_ships()
engine.to_csv('prod_metadata.csv')
