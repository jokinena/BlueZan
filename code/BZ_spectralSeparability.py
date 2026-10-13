# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:36:49 2026

Zanzibar spectral gap analysis

@author: E1008409
"""


import argparse
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def jeffries_matusita_distance(X1, X2, regularization=1e-6):
    """
    Compute Jeffries-Matusita distance between two sets of observations.

    Parameters:
        X1, X2 : array-like, shape (n_samples, n_features)
            Samples for each class.
        regularization : float
            Small value added to diagonal of covariance matrices for stability.

    Returns:
        jm : float
            Jeffries-Matusita distance (range: 0 to 2).
    """
    X1 = np.asarray(X1)
    X2 = np.asarray(X2)

    # Mean vectors
    mu1 = np.mean(X1, axis=0)
    mu2 = np.mean(X2, axis=0)

    # Covariance matrices
    cov1 = np.cov(X1, rowvar=False)
    cov2 = np.cov(X2, rowvar=False)

    # Regularization to avoid singular matrices
    cov1 += regularization * np.eye(cov1.shape[0])
    cov2 += regularization * np.eye(cov2.shape[0])

    # Average covariance
    cov_avg = 0.5 * (cov1 + cov2)

    # Difference of means
    diff = (mu1 - mu2).reshape(-1, 1)

    # Inverse of average covariance
    try:
        inv_cov_avg = np.linalg.inv(cov_avg)
    except np.linalg.LinAlgError:
        inv_cov_avg = np.linalg.pinv(cov_avg)

    # Bhattacharyya distance components
    term1 = 0.125 * (diff.T @ inv_cov_avg @ diff).item()

    det_cov1 = np.linalg.det(cov1)
    det_cov2 = np.linalg.det(cov2)
    det_cov_avg = np.linalg.det(cov_avg)

    # Avoid log(0)
    if det_cov1 <= 0 or det_cov2 <= 0 or det_cov_avg <= 0:
        term2 = 0.0
    else:
        term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))

    B = term1 + term2

    # Jeffries-Matusita distance
    jm = 2 * (1 - np.exp(-B))

    return jm

def jm_distance_geodataframe(gdf, feature_cols, class_col, regularization=1e-6):
    """
    Compute Jeffries-Matusita distance between all class pairs in a GeoDataFrame.

    Parameters:
        gdf : GeoDataFrame
        feature_cols : list of str
            Columns used as features
        class_col : str
            Column with class labels
        regularization : float
            Stability term for covariance matrices

    Returns:
        results : dict
            {(class1, class2): JM distance}
    """
    results = {}

    # Get unique classes
    classes = gdf[class_col].dropna().unique()

    # Loop through all class pairs
    for c1, c2 in combinations(classes, 2):
        X1 = gdf[gdf[class_col] == c1][feature_cols].values
        X2 = gdf[gdf[class_col] == c2][feature_cols].values

        # Skip if not enough samples
        if len(X1) < 2 or len(X2) < 2:
            results[(c1, c2)] = np.nan
            continue

        # Means
        mu1 = np.mean(X1, axis=0)
        mu2 = np.mean(X2, axis=0)

        # Covariances
        cov1 = np.cov(X1, rowvar=False)
        cov2 = np.cov(X2, rowvar=False)

        # Regularization
        cov1 += regularization * np.eye(cov1.shape[0])
        cov2 += regularization * np.eye(cov2.shape[0])

        cov_avg = 0.5 * (cov1 + cov2)

        diff = (mu1 - mu2).reshape(-1, 1)

        # Inverse covariance
        try:
            inv_cov_avg = np.linalg.inv(cov_avg)
        except np.linalg.LinAlgError:
            inv_cov_avg = np.linalg.pinv(cov_avg)

        # Bhattacharyya terms
        term1 = 0.125 * (diff.T @ inv_cov_avg @ diff).item()

        det_cov1 = np.linalg.det(cov1)
        det_cov2 = np.linalg.det(cov2)
        det_cov_avg = np.linalg.det(cov_avg)

        if det_cov1 <= 0 or det_cov2 <= 0 or det_cov_avg <= 0:
            term2 = 0.0
        else:
            term2 = 0.5 * np.log(det_cov_avg / np.sqrt(det_cov1 * det_cov2))

        B = term1 + term2

        # JM distance
        jm = 2 * (1 - np.exp(-B))

        results[(c1, c2)] = jm

    return results

def sampleRaster(raster_fp, geodataframe):
        
    # sample coords
    with rio.open(raster_fp) as src:
        profile = src.profile
        crs = src.crs.to_epsg()
        # check crs
        if geodataframe.crs != src.crs:
            geodataframe = geodataframe.to_crs(src.crs)
        # check geometry, explode if MultiPoint
        if geodataframe.geometry.geom_type.str.contains('MultiPoint').any() == True:
            sp = geodataframe.geometry.explode()
            coords = [(x,y) for x,y in zip(sp.geometry.x, sp.geometry.y)]
        else:
            # get point coords
            coords = [(x,y) for x,y in zip(geodataframe.geometry.x, geodataframe.geometry.y)]
        # sample
        geodataframe['sampled'] = [x for x in src.sample(coords)]
    # output columns from number of bands
    colnames = ['Band'+str(n) for n in np.arange(1,profile['count']+1)]
    # extract sampled
    geodataframe[colnames] = gpd.GeoDataFrame(geodataframe.sampled.tolist(), index=geodataframe.index)
    
    return geodataframe, colnames

# files
fp_img = '/mnt/d/users/e1008409/MK/sansibar/S2/Chwaka/S2_2018_chwaka.tif'
fp_kmeans = '/mnt/d/users/e1008409/MK/sansibar/012026/Chwaka/KMeans/S2_2018_15kmeans.tif'
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/012026/Chwaka/KMeans/KMeans15_poly_strat_points_30_seed123.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BlueZan_habitat_data_11_2024_labels.gpkg'
fp = "/mnt/d/users/e1008409/MK/DNASense/FIN/Finland_habitat_data_ml_2016-2023_v5.gpkg"
# read
with rio.open(fp_img) as src:
    img = src.read()
    profile = src.profile

with rio.open(fp_kmeans) as src:
    kmeans = src.read()
    kmeans_profile = src.profile

gdf = gpd.read_file(fp, engine='pyogrio')
# select by date
start_date = pd.to_datetime('2016-1-1', format="%Y-%m-%d")
end_date = pd.to_datetime('2019-12-31', format="%Y-%m-%d")
start_date = 2016
end_date = 2019 
gdf = gdf[(gdf['YEAR'] >= start_date) & (gdf['YEAR'] <= end_date)]
gdf_out = fp.split('_2')[0] + '_2016-2019_32634.gpkg'
gdf = gdf.to_crs('EPSG:32634')
gdf.to_file(gdf_out, engine='pyogrio')
# drop gdf points with DN = 0, equals nodata
gdf = gdf[gdf['DN'] != 0]
# sample
gdf, cols = sampleRaster(fp_img, gdf)

# compute jeffries matusita distance
jmdist = jm_distance_geodataframe(gdf, cols, 'subs_highest')

# update classes in kmeans raster based on sampled jmdist values
jmdist_thr = 1.8
for i in jmdist.keys():
    print(i)
    if jmdist.get(i) < jmdist_thr:
        kmeans = np.where(kmeans == i[0], i[1], kmeans)

# save updated kmeans raster
kmeans_out = os.path.join(os.path.dirname(fp_kmeans), os.path.basename(fp_kmeans).split('.')[0] + '_jm.tif')
with rio.open(kmeans_out, 'w', **kmeans_profile) as dst:
    dst.write(kmeans)




















