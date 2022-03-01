# Processing Sentinel-1 radar images

Minimal code for classification and object detection tasks.

# Quickstart

Init the engine:
```python
from core import SatEngine

# Init the list of downloaded products.
engine = SatEngine('username', 'password',
    dir_downloads = '/path/to/downloads',
    dir_products = '/path/to/data',
    dir_images = '/path/to/images')
```

Load metadata from CSV:
```python
engine.from_csv('path/to/metadata.csv')
```

Query Copernicus and download products:
```python
fn_geojson = 'aoi.geojson'
date_start = '20210101'
date_end = '20211231'

engine.download(fn_geojson, date_start, date_end, policy = 'Contains',
  check_only = False, query = False)
```

Extract AOI and detect ships:
```python
engine.extract_aoi(fn_geojson)
res = engine.detect_ships()
```

Save metadata (including detection results) to CSV:
```python
engine.to_csv('path/to/metadata.csv')
```
