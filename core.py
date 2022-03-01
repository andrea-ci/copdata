# -*- coding: utf-8 -*-
import gc
from os import path, makedirs, listdir
import json
import csv
import zipfile
import numpy as np
import cv2 as cv
import tqdm
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from sentinelsat.exceptions import LTATriggered
import geoutils as gt
import detection as det
import utils

apihub_url = 'https://apihub.copernicus.eu/apihub'

class ProductStore:
    """Manages metadata for Sentinel-1 products."""

    def __init__(self):

        self._all_metadata = []

    def to_csv(self, filename):
        """Saves metadata to CSV file."""

        if len(self.metadata) == 0:
            # Nothing to save
            return

        field_names = list(self.metadata[0].keys())

        with open(filename, 'w', encoding = 'utf-8') as file:

            csv_writer = csv.DictWriter(file, fieldnames = field_names,
                delimiter = ':', quotechar = '"', quoting = csv.QUOTE_MINIMAL)

            csv_writer.writeheader()

            for meta in self._all_metadata:

                # Serialize list of polygon coordinates.
                meta['polygon'] = json.dumps(meta['polygon'])

                csv_writer.writerow(meta)

    def from_csv(self, filename):

        with open(filename, 'r', encoding = 'utf-8') as file:

            csv_reader = csv.DictReader(file, delimiter = ':',
                quotechar = '"', quoting = csv.QUOTE_MINIMAL)

            for meta in csv_reader:

                if meta['isDownloaded'] == 'True':
                    meta['isDownloaded'] = True
                else:
                    meta['isDownloaded'] = False

                if meta['aoiExtracted'] == 'True':
                    meta['aoiExtracted'] = True
                else:
                    meta['aoiExtracted'] = False

                # Deserialize list of polygon coordinates.
                meta['polygon'] = json.loads(meta['polygon'])

                self._all_metadata.append(meta)

    def add_metadata(self, product_id, raw_metadata):
        """Adds a new record to the CSV file."""

        # Extract metadata of current product.
        title = raw_metadata['title']
        ingestion_date = raw_metadata['Ingestion Date']
        data_folder = raw_metadata['Filename']
        pass_direction = raw_metadata['Pass direction']
        img_footprint = raw_metadata['footprint']

        # Compose names of folders.
        zip_file = title + '.zip'

        # Parse the polygon of Footprint.
        poly = self.parse_polygon(img_footprint)

        # Compose the string of date.
        month_str = str(ingestion_date.month)
        if len(month_str) == 1:
            month_str = '0' + month_str
        day_str = str(ingestion_date.day)
        if len(day_str) == 1:
            day_str = '0' + day_str
        date_str = f'{ingestion_date.year}{month_str}{day_str}'

        pass_direction = pass_direction.lower()

        metadata = {
            'productId': product_id,
            'zipFile': zip_file,
            'dataFolder': data_folder,
            'ingestionDate': ingestion_date,
            'passDirection': pass_direction,
            'polygon': poly,
            'isDownloaded': False,
            'aoiExtracted': False
        }

        self._all_metadata.append(metadata)

    @property
    def metadata(self):
        """Returns metadata of products."""

        return self._all_metadata

    def parse_polygon(self, polygon_str):
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

class SatEngine(ProductStore):
    """Downloads data and processes Sentinel-1 images."""

    def __init__(self, username, password, dir_downloads = 'downloads',
        dir_products = 'data', dir_images = 'images'):

        super().__init__()

        # Create folder for the product images.
        if not path.exists(dir_downloads):
            makedirs(dir_downloads)
        # Create folder for the product images.
        if not path.exists(dir_products):
            makedirs(dir_products)
        # Create folder for the product images.
        if not path.exists(dir_images):
            makedirs(dir_images)

        # Set paths for Sentinel products and images.
        self.dir_downloads = dir_downloads
        self.dir_products = dir_products
        self.dir_images = dir_images

        self.aoi = None

        # Init Sentinel API.
        self.api = SentinelAPI(username, password, apihub_url)

    def download(self, fn_geojson, date_start, date_end, check_only = False,
        query = True, policy = 'Contains'):
        """Downloads GRD High-Resolution images from Sentinel-1."""

        if query is True:

            # Read GEOJSON and convert to WKT format.
            json_data = read_geojson(fn_geojson)
            footprint = geojson_to_wkt(json_data)

            # Query Copernicus API for products.
            products = self.api.query(footprint, date = (date_start, date_end),
                platformname = 'Sentinel-1', producttype = 'GRD',
                area_relation = policy)

            # Convert to DataFrame.
            df_products = self.api.to_dataframe(products)
            print(f'Found {df_products.shape[0]} products from Copernicus.')

            # Filter products and update metadata.
            for product_id in tqdm(df_products.index):

                metadata = self.api.get_product_odata(product_id, full = True)

                if metadata['Mode'] == 'IW' and metadata['Resolution'] == 'High':
                    self.add_metadata(product_id, metadata)

        # Download/extract the products.
        if check_only is False:

            for meta in self.metadata:

                product_id = meta['productId']

                if path.isfile(path.join(self.dir_downloads, meta['zipFile'])):
                    print('Product already downloaded.')

                else:
                    print(f'Downloading product {product_id}.')

                    try:
                        self.api.download(product_id, directory_path = self.dir_downloads)
                        meta['isDownloaded'] = True

                    except LTATriggered:
                        print('Long Term Archive triggered, cannot download right now.')
                        continue

                    except Exception as err:
                        print(f'Error occurred: {str(err)}, skipping this product.')
                        continue

                if path.isdir(path.join(self.dir_products, meta['dataFolder'])):
                    print('Data folder already extracted.')

                else:
                    zip_file = meta['zipFile']
                    print(f'Extracting {zip_file} to data folder.')

                    with zipfile.ZipFile(path.join(self.dir_downloads, zip_file), 'r') as zip_ref:
                        zip_ref.extractall(self.dir_products)

    def extract_aoi(self, fn_geojson):
        """Crops the image and extracts the AOI."""

        # Extracts bbox from GEOJSON.
        json_data = read_geojson(fn_geojson)
        geometry = json_data['features'][0]['geometry']
        poly_aoi = geometry['coordinates'][0]
        longs = [pnt[0] for pnt in poly_aoi]
        lats = [pnt[1] for pnt in poly_aoi]
        lat_min = min(lats)
        lat_max = max(lats)
        long_min = min(longs)
        long_max = max(longs)
        bbox = (long_min, lat_min, long_max, lat_max)

        for meta in self.metadata:

            product_id = meta['productId']
            data_folder = meta['dataFolder']
            poly = meta['polygon']
            direction = meta['passDirection']

            print(f'Extracting AOI from product {product_id}.')

            if meta['isDownloaded'] is not True:
                continue

            # Create folder for the product images.
            if not path.exists(path.join(self.dir_images, product_id)):
                makedirs(path.join(self.dir_images, product_id))

            # Load VV/VH images.
            path_image_vv = None
            path_image_vh = None

            path_images = path.join(self.dir_products, data_folder, 'measurement')

            for file in listdir(path_images):

                filepath = path.join(path_images, file)

                if '-vv-' in filepath:
                    path_image_vv = filepath
                elif '-vh-' in filepath:
                    path_image_vh = filepath

            image_vv = cv.imread(path_image_vv, cv.IMREAD_UNCHANGED)
            image_vh = cv.imread(path_image_vh, cv.IMREAD_UNCHANGED)
            n_rows, n_cols = image_vv.shape

            # Adjust the image according to the pass direction.
            if direction == 'descending':
                # Flip along East-West axis.
                image_vv = image_vv[:, ::-1]
                image_vh = image_vh[:, ::-1]
            else:
                # Flip along North-South axis.
                image_vv = image_vv[::-1, :]
                image_vh = image_vh[::-1, :]

            # Resize and save a preview.
            resized_vv = image_vv[::20, ::20].astype(np.float64)
            resized_vh = image_vh[::20, ::20].astype(np.float64)
            resized_rgb = utils.create_image_rgb(resized_vv, resized_vh)
            fn = path.join(self.dir_images, product_id, 'rgb_small.png')
            cv.imwrite(fn, resized_rgb[:, :, ::-1])

            # Free up saved images.
            del resized_vv, resized_vh, resized_rgb
            gc.collect()

            # Evaluate the polygon and sort bottom/upper right points.
            sorted_by_lat = sorted(poly, key = lambda x : x[1])
            bottom_right = sorted(sorted_by_lat[:2], key = lambda x : x[0])[1]
            upper_right = sorted(sorted_by_lat[2:], key = lambda x : x[0])[1]

            # Angle is negated because we must perform the opposite transformation
            # between coordinate systems, i.e. from swath-based to NS-EW.
            theta = - gt.get_bearing(bottom_right, upper_right)
            print(f'Swath bearing is {-theta}.')

            # Evaluate the projection of image rows/cols along North/East.
            dE_col, dN_col = gt.rotate_spacing(10, 0, theta)
            dE_row, dN_row = gt.rotate_spacing(0, -10, theta)

            print(f'Projection of image columns to North and East: {dE_col}m, {dN_col}m')
            print(f'Projection of image rows to North and East: {dE_row}m, {dN_row}m')

            img_coords = gt.gen_pixel_coords(poly, n_rows, n_cols, dN_col, dE_col,
                dN_row, dE_row)

            print('Coordinates at corners:')
            print(img_coords[0, 0, :])
            print(img_coords[0, n_cols - 1, :])
            print(img_coords[n_rows - 1, 0, :])
            print(img_coords[n_rows - 1, n_cols - 1, :])

            r_min, r_max, c_min, c_max = gt.find_bbox_inds(img_coords, bbox)
            print(r_min, r_max, c_min, c_max)

            fn_cropped_vv = path.join(self.dir_images, product_id, 'cropped_vv.png')
            fn_cropped_vh = path.join(self.dir_images, product_id, 'cropped_vh.png')
            cropped_vv = image_vv[r_min : r_max + 1, c_min : c_max + 1]
            cropped_vh = image_vh[r_min : r_max + 1, c_min : c_max + 1]
            cv.imwrite(fn_cropped_vv, cropped_vv)
            cv.imwrite(fn_cropped_vh, cropped_vh)

            cropped_vv = cropped_vv.astype(np.float64)
            cropped_vh = cropped_vh.astype(np.float64)
            image_rgb = utils.create_image_rgb(cropped_vv, cropped_vh)
            fn = path.join(self.dir_images, product_id, 'cropped.png')
            cv.imwrite(fn, image_rgb[:, :, ::-1])

            # Free up saved images.
            del cropped_vv, cropped_vh, image_rgb
            del image_vv, image_vh
            gc.collect()

            meta['aoiExtracted'] = True

    def detect_ships(self):
        """Detects ships on images."""

        for meta in self.metadata:

            product_id = meta['productId']
            print(f'Starting to process product {product_id}.')

            path_cropped_vv = path.join(self.dir_images, product_id, 'cropped_vv.png')
            path_cropped_vh = path.join(self.dir_images, product_id, 'cropped_vh.png')

            image_vv = cv.imread(path_cropped_vv, cv.IMREAD_UNCHANGED)
            image_vv = image_vv.astype(np.float64)
            image_vh = cv.imread(path_cropped_vh, cv.IMREAD_UNCHANGED)
            image_vh = image_vh.astype(np.float64)

            print('Starting CFAR detection for VV image.')
            targets_vv = det.cfar_detector(image_vv, size_outer = 100,
                size_inner = 40, thr = 4.7)
            print(f'Found {np.sum(targets_vv)} positive samples.')

            print('Starting CFAR detection for VH image.')
            targets_vh = det.cfar_detector(image_vh, size_outer = 100,
                size_inner = 40, thr = 4.7)
            print(f'Found {np.sum(targets_vh)} positive samples.')

            # Fuse targets from both polarizations.
            targets = np.zeros(targets_vv.shape, dtype = np.uint8)
            targets[(targets_vv == 1) & (targets_vh == 1)] = 1
            n_targets = np.sum(targets)
            print(f'Found {n_targets} common targets.')

            print('Finding target contours.')
            contours, hierarchy = cv.findContours(targets, cv.RETR_TREE,
                cv.CHAIN_APPROX_SIMPLE)

            # Convert contours into boxes.
            boxes = det.contours_to_boxes(contours)

            # Merge overlapping boxes.
            merged_boxes = det.merge_overlapping_boxes(boxes)
            n_ships = len(merged_boxes)
            print(f'Found {n_ships} non overlapping shapes.')

            # Add the ship boxes to the RGB image.
            fn_cropped = path.join(self.dir_images, product_id, 'cropped.png')
            fn_detection = path.join(self.dir_images, product_id, 'detections.png')
            img = cv.imread(fn_cropped)
            det.label_image_with_boxes(fn_detection, img, merged_boxes)

            meta['numShips'] = n_ships

        return self.metadata
