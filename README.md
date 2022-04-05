# Library for accessing and pre-processing Sentinel-1 and Sentinel-2 products

Minimal code for classification and object detection tasks.

# Quickstart

Query products from Copernicus:

```python
from sentinelsat import read_geojson, geojson_to_wkt
from copdata.copernicus import CopernicusClient
from copdata.core import SentinelProduct

# Init the Copernicus client.
cc = CopernicusClient('username', 'password')

# Read geojson and convert to WKT format.
fn_geojson = 'aoi.geojson'
json_data = read_geojson(fn_geojson)
aoi_wkt = geojson_to_wkt(json_data)

# Query for products.
date_start = '20211201'
date_end = '20211231'
product_ids = cc.query_products(aoi_wkt, date_start, date_end, details = False)
```

Download a product from Copernicus:

```python

metadata = cc.download_product(product_id, include_data = True)
```

Load a product already downloaded and extracted:

```python

from copdata.utils import create_image_rgb

# Extract the AOI coordinates.
poly_aoi = json_data['features'][0]['geometry']['coordinates'][0]

# Open the product.
sd = SentinelProduct(metadata)

# Check the size of the image(s).
print(sd.img_shape)

# Extracts the area of interest from the product (Sentinel-1).
cropped_vv, cropped_vh = sd.extract_aoi(poly_aoi)

# Create a pseudo-RGB image of AOI.
cropped_rgb = create_image_rgb(cropped_vv, cropped_vh)

```
