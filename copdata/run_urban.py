# -*- coding: utf-8 -*-
"""This script perform a query to Copernicus, filters and downloads valid products
and extracts to their SAFE folders. """
import pickle
from core import SatEngine
import numpy as np
import cv2 as cv
from classifiers import map_classification
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from medpy.filter.smoothing import anisotropic_diffusion

#color_codes = np.array([[255,0,0],[0,255,0],[0,0,255],[0,255,255]])
color_codes = np.array([
 	(0,0,0),
 	(255,255,255),
 	(255,0,0),
 	(0,255,0),
 	(0,0,255),
 	(255,255,0),
 	(0,255,255),
 	(255,0,255),
 	(192,192,192),
 	(128,128,128),
 	(128,0,0),
 	(128,128,0),
 	(0,128,0),
 	(128,0,128),
 	(0,128,128),
 	(0,0,128)
])

# tci_roma.png - resized
# coords: x,y
training_metadata = [
    {
        'name' : 'water',
        'label' : 0,
        'pixel_coords': [
            (123, 1263),
            (447, 1316),
            (797, 1887),
            (1247, 2299),
            (1681, 2119)
        ]
    },
    {
        'name' : 'vegetation',
        'label' : 1,
        'pixel_coords': [
            (1434, 2084),
            (1630, 2149),
            (2465, 2438),
            (896, 1839),
            (2342, 2518)
        ]
    },
    {
        'name' : 'terrain',
        'label' : 2,
        'pixel_coords': [
            (1417, 2298),
            (1757, 2204),
            (1370, 1869),
            (2370, 2022),
            (2958, 2389)
        ]
    },
    {
        'name' : 'urban_land',
        'label' : 3,
        'pixel_coords': [
            (1819, 2072),
            (1584, 2007),
            (1877, 2380),
            (2893, 1376),
            (2913, 1381)
        ]
    }
]

fn_in = 'tci_roma.png'

img = cv.imread(fn_in, cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
n_rows, n_cols = img.shape[:2]
print(f'Image has size {n_rows}x{n_cols}.')

X = []
y = []
for tm in training_metadata:

    for coords in tm['pixel_coords']:

        rgb_values = list(img[coords[1], coords[0], :])
        X.append(rgb_values)
        y.append(tm['label'])

training_data = X, y

with open('training_data.pickle', 'wb') as file:
    pickle.dump(training_data, file)
exit()
# Resize and save image.
#img = img[4800:7500, 6000:9700, :]
#cv.imwrite(fn_in.replace('.jp2', ('.png')), img[:,:,::-1])

if 0:
    print('Starting K-Neighbours classification.')
    clf = KNeighborsClassifier(n_neighbors = 3)

else:
    print('Starting Random Forest classification.')
    clf = RandomForestClassifier(n_estimators = 10, max_depth = 2,
        random_state = 0)

# Fit the estimator.
clf.fit(X, y)

# Classify the image.
fn_in = 'tci_seattle.png'
img = cv.imread(fn_in, cv.IMREAD_COLOR)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
n_rows, n_cols = img.shape[:2]
print(f'Image has size {n_rows}x{n_cols}.')

flat_image = np.reshape(img, [-1,3])
predicted = clf.predict(flat_image)
mask = np.reshape(predicted, (n_rows, n_cols))

pixel_probs = clf.predict_proba(flat_image) # NxM,4
class_probs = np.reshape(pixel_probs, (n_rows, n_cols, 4)) # N,M,4

# Reduce classes to urban vs. non-urban.
mask[mask == 1] = 0
mask[mask == 2] = 0
mask[mask == 3] = 1

mask_rgb = color_codes[mask.astype(int)]
fn_out = fn_in.replace('.png', f'_raw_mask.png')
cv.imwrite(fn_out, mask_rgb[:, :, ::-1])

print('Starting anisotropic filtering.')

class_probs_flt = anisotropic_diffusion(class_probs, niter = 1, kappa = 50,
    gamma = 0.1, option = 1)
mask_flt = np.argmax(class_probs_flt, axis = 2)

# Reduce classes to urban vs. non-urban.
mask_flt[mask_flt == 1] = 0
mask_flt[mask_flt == 2] = 0
mask_flt[mask_flt == 3] = 1

mask_rgb = color_codes[mask_flt.astype(int)]

print('Saving the image.')
fn_out = fn_in.replace('.png', f'_mask.png')
cv.imwrite(fn_out, mask_rgb[:, :, ::-1])
exit()



kmeans = KMeans(n_clusters = n_clusters, random_state = 0).fit(flat_image)
mask = np.reshape(kmeans.labels_, (n_rows, n_cols))
print(mask.shape)
#mask = map_classification(img_vv, err_tol = 0.01, n_classes = 2, scale = 5,
#    max_iters = 5000)

mask_rgb = color_codes[mask.astype(int)]
cv.imwrite(fn_out, mask_rgb[::-1])
exit()

# Init the list of downloaded products.
engine = SatEngine('andrea32874763', 'yigXUR6M!eunDdv',
    dir_downloads = '/hdd2/sentinel-products/downloads',
    dir_products = '/hdd2/sentinel-products/data',
    dir_images = '/hdd2/sentinel-products/images')

engine.add_product('041a25c2-ed40-4295-9651-472284ea39bf')
engine.download_products()
engine.to_csv('roma.csv')

fn_geojson = 'geoareas/naples.geojson'
date_start = '20211201'
date_end = '20211231'

engine.query(fn_geojson, date_start, date_end, policy = 'Contains')
