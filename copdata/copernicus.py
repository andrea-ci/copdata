# -*- coding: utf-8 -*-
import pandas as pd
from sentinelsat import SentinelAPI
from sentinelsat.exceptions import LTATriggered
from . import utils

apihub_url = 'https://apihub.copernicus.eu/apihub'

# Init the logger.
logger = utils.get_logger()

class CopernicusClient:
    """Gets information on Sentinel products from Copernicus HUB API."""

    def __init__(self, username, password, dir_downloads = '.'):

        # Initialize the client for Sentinel API.
        self.api = SentinelAPI(username, password, apihub_url)
        self.dir_downloads = dir_downloads

    def query_products(self, footprint, date_start, date_end, details = False,
        mission = 'sentinel-1', policy = 'Contains', downsampling = None):
        """Query products based on area and time range and adds metadata
        of valid products."""

        res = []

        # Query Copernicus API for products.
        products = self.api.query(footprint, date = (date_start, date_end),
            platformname = 'Sentinel-1', producttype = 'GRD',
            area_relation = policy)

        # Convert to DataFrame.
        df_products = self.api.to_dataframe(products)

        if downsampling is not None:
            # Downsample the products on a date basis.
            df_products['Date'] = pd.to_datetime(df_products.ingestiondate)
            resampler = df_products.resample(downsampling, on = 'Date')
            df_products = resampler.first().drop('Date', axis = 1)
            df_products.index = df_products.uuid.values

        # Filter products and update metadata.
        for product_id in df_products.index:

            raw_metadata = self.api.get_product_odata(product_id, full = True)

            if raw_metadata['Mode'] == 'IW' and raw_metadata['Resolution'] == 'High':
                metadata = self._parse_metadata(product_id, raw_metadata)

                if details is True:
                    res.append(metadata)
                else:
                    res.append(product_id)

        return res

    def download_products(self, product_ids, include_data = True):
        """Downloads multiple products by ID."""

        res = []
        for product_id in product_ids:

            meta = self.download_product(product_id, include_data = include_data)
            res.append(meta)

        return res

    def download_product(self, product_id, include_data = True):
        """Downloads a product by ID."""

        logger.info(f'Downloading product {product_id}.')

        is_downloaded = False

        try:
            raw_metadata = self.api.get_product_odata(product_id, full = True)

            if include_data is True:
                self.api.download(product_id, directory_path = self.dir_downloads)

            is_downloaded = True

        except LTATriggered:
            logger.warn('Triggering the Long Term Archive, product cannot be downloaded right now.')

        except Exception as err:
            logger.error(f'Error occurred: {str(err)}, skipping this product.')

        if is_downloaded is True:
            metadata = self._parse_metadata(product_id, raw_metadata)
        else:
            metadata = None

        return metadata

    def _parse_metadata(self, product_id, raw_metadata):
        """Adds a new record to the CSV file."""

        # Extract metadata of current product.
        title = raw_metadata['title']
        ingestion_date = raw_metadata['Ingestion Date']
        data_folder = raw_metadata['Filename']
        pass_direction = raw_metadata['Pass direction']
        img_footprint = raw_metadata['footprint']
        mission = raw_metadata['Satellite'].lower()

        # Compose names of folders.
        zip_file = title + '.zip'

        # Parse the polygon of Footprint.
        poly = self._parse_polygon(img_footprint)

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
            'mission': mission,
            'zipFile': zip_file,
            'dataFolder': data_folder,
            'ingestionDate': date_str,
            'passDirection': pass_direction,
            'polygon': poly,
            'isDownloaded': False,
            'aoiExtracted': False
        }

        return metadata

    def _parse_polygon(self, polygon_str):
        """
        Parses the polygon string obtained by Copernicus product metadata.
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
