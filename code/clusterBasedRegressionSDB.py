# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:00:48 2023

Satellite derived bathymetry using cluster based regression
Geyman & Maloof 2019
https://doi.org/10.1029/2018EA000539

Steps
1. K-means clustering
2. Log band ratios (Stumpf et al.)

@author: E1008409
"""

import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from rasterstats import zonal_stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
    
# function to compute logarithmic band ratio from given bands
def StumpfLogBandRatio(image_array, bands):
    """
    Compute logarithmic band ratio as ln(1000*band) / ln(1000*band), see Stumpf et al. (2003)

    Parameters
    ----------
    image_array : np.array
        Image array
    bands : tuple or list
        Band numbers 

    Returns
    -------
    logbr : np.array
        Log-ratio of image bands

    """
    logbr = (np.log(1000*image_array[bands[0]])) / (np.log(1000*image_array[bands[1]]))
    logbr = logbr.reshape((logbr.shape[0], logbr.shape[1]))
    
    return logbr

def getRioBbox(filepath):
    """
    Get bounding box of a raster layer 

    Parameters
    ----------
    filepath : str
        String filepath.

    Returns
    -------
    Bounding box as GeoDataFrame and JSON format list of bounding box coordinates

    """
    import rasterio as rio
    import geopandas as gpd
    from shapely.geometry import box
    from fiona.crs import from_epsg
    import json
    
    with rio.open(filepath) as src:
        bbox = box(src.bounds[0],src.bounds[1],src.bounds[2],src.bounds[3]) # left bottom, right top
     # polygonize bbox
    bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs) 
    # get features 
    coords = [json.loads(bbox_poly.to_json())['features'][0]['geometry']]
    return bbox_poly, coords

def clipToBbox(raster_fp, vector_fp):
    """
    Clip to bounding box of raster layer

    Parameters
    ----------
    raster_fp : TYPE
        DESCRIPTION.
    vector_fp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    bbox, coords = getRioBbox(raster_fp)
    
    # read vector
    gdf = gpd.read_file(vector_fp)
    # reproject bbox if not same with gdf
    if bbox.crs != gdf.crs:
        bbox = bbox.to_crs(gdf.crs)
    # clip
    clipped = gpd.clip(gdf, bbox, keep_geom_type=True)
    return clipped

def maskSigma3Outliers(array):
    """
    Mask outliers > 3 *std from the mean

    Parameters
    ----------
    array : TYPE
        DESCRIPTION.

    Returns
    -------
    arr_out : TYPE
        DESCRIPTION.

    """
    a_mean = np.nanmean(array)
    a_std = np.nanstd(array)
    sigthree = 3*a_std
    arr_out = np.where(np.abs((array - a_mean)) > sigthree, np.nan, array)
    return arr_out

# filepaths
fp = '/mnt/d/users/e1008409/MK/sansibar/S2/downloads/S2_masked/S2B_MSIL2A_20260723T072619_N0512_R049_T37MEP_20260723T112037_B02B03B04B08_masked.tif'
# ground truth
fp_gt = "/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BlueZan_habitat_data_11_2024.gpkg"

# ------------------------ # 
# 1. K-means unsupervised classification
# ------------------------ #

# open
with rio.open(fp) as src:
    img = src.read()
    meta = src.meta
# nodatamask
mask = np.isnan(img[0])

# flatten each band
flat_data = np.empty((img.shape[1]*img.shape[2], img.shape[0]))

for i in range(img.shape[0]):
    band = img[i,:,:]
    flat_data[:,i-1] = band.flatten()

# standard scaling
#scaler = StandardScaler().fit(flat_data)
#scaled_img = scaler.transform(flat_data)

# replace nan
scaled_img = np.where(np.isnan(flat_data), -9999, flat_data)

# kmeans clustering
km = KMeans(n_clusters=8)
km.fit(scaled_img)
pred = km.predict(scaled_img)
cluster_centers = km.cluster_centers_
# plot clusters
fig, ax = plt.subplot()
ax.plot()


# back to 2-d array
clustered = pred.reshape(src.shape[0], src.shape[1])
plt.imshow(clustered)

# mask nodata area
clustered = np.where(mask == True, -9999, clustered)

# save
unsup_out = os.path.join(os.path.dirname(fp), 'n6_clustered.tif')
unsup_meta = meta.copy()
unsup_meta.update(dtype='int32',
                  nodata=-9999,
                  count=1)

with rio.open(unsup_out, 'w', **unsup_meta) as dst:
    dst.write(clustered.astype(rio.int16), 1)


# --------------------------- #
# 2. Log band ratios

# list of unique band pairs
unique_pairs = list(itertools.combinations(np.arange(1,4), 2))
#unique_pairs = unique_pairs[1:]
# compute band ratios
band_ratios = [StumpfLogBandRatio(img, u) for u in unique_pairs]
# to array
band_ratios = np.array(band_ratios)
# transpose
#band_ratios = np.transpose(band_ratios, (1,2,0))

# save stack for sampling
outdir = os.path.join(os.path.dirname(fp), 'sdb')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
stack_out = os.path.join(outdir, 'band_ratios_temp.tif')

stack_meta = meta.copy()
stack_meta.update(count=band_ratios.shape[0],
                  nodata=np.nan,
                  dtype='float32')

with rio.open(stack_out, 'w', **stack_meta) as dst:
    dst.write(band_ratios)

# ----------------------------- # 
# 3. Read field data and prepare dataset

#gdf = clipToBbox(fp, fp_gt)
# columns to select
#cols = ['id', 'menetelma', 'syvyys_mitattu', 'geometry']
# select
#gdf = gdf[cols]


# read data
gdf = gpd.read_file(fp_gt) # layer='training2_sand_soft_hard_class'
# select relevant columns
gdf = gdf[['id', 'menetelma', 'syvyys_mitattu', 'savcov', 'geometry']]

# drop na string rows in menetelma
gdf = gdf[gdf['menetelma'] != 'NA']
# to int
gdf.menetelma = gdf.menetelma.astype(int)

# fix depths to floats
if gdf.syvyys_mitattu.dtype != 'float64':
    # convert syvyys_mitattu to numeric
    gdf['temp'] = gdf.syvyys_mitattu.str.split('-')
    gdf[['temp', 'temp2']] = gpd.GeoDataFrame(gdf.temp.tolist(), index=gdf.index)
    gdf['field_depth'] = gdf['temp2'].str.replace(',','.').astype(float)
    # drop temp columns
    gdf = gdf.drop(['temp', 'temp2'], axis=1)

else:
    # absolute values
    gdf['field_depth'] = abs(gdf.syvyys_mitattu)

#gdf_dropvid = gdf.loc[gdf.menetelma == 42] # for testing
# select dive, aquascope, drop video points
gdf = gdf[gdf.menetelma.isin([21, 23, 25, 42])]

# keep rows where field depth > 0.1
gdf = gdf[gdf.field_depth >= 0.1]

######### sampling ###########

# check crs
src = rio.open(stack_out)
if gdf.crs.to_epsg() != src.crs.to_epsg():
    print('Reprojecting points')
    gdf = gdf.to_crs(epsg=src.crs.to_epsg())
src.close()

# create buffer 
gdf['poly_geom'] = gdf.geometry.buffer(2, cap_style=3)
# band ratio column names
br_cols = ['bg', 'by', 'br',
     'gy', 'gr', 'yr']

# rasterstats
n=0
for i in band_ratios:
    stats = zonal_stats(gdf.poly_geom, i, stats=['mean'], affine=stack_meta['transform'], all_touched=True, nodata=stack_meta['nodata'])
    statlist = [d['mean'] for d in stats]
    gdf[br_cols[n]] = statlist
    n+=1

# sample cluster
# read
with rio.open(unsup_out) as src:
    clustered = src.read()
    clust_meta = src.meta
clusters = zonal_stats(gdf.geometry, clustered[0], stats=['mean'], affine=clust_meta['transform'], all_touched=False, nodata=clust_meta['nodata'])
clusters = [c['mean'] for c in clusters]
gdf['cluster'] = clusters

# drop if sampled na
gdf = gdf.dropna()
gdf = gdf[gdf.cluster != clust_meta['nodata']]

# drop depths > 6m
gdf = gdf[gdf.field_depth <= 6.0]

# plot
fig, ax = plt.subplots(2,3)
i,j = 0,0
for col in br_cols:
    ax[i,j].scatter(gdf[col], gdf.field_depth, s=0.5, color='Black')
#    ax[i,j].set_xticks(np.arange(0,max(gdf[col]), 0.25))
    ax[i,j].set_yticks(np.arange(0,max(gdf.field_depth), 1))
    ax[i,j].set_title(col)
    i += 1
    if i > 1:
        i = 0
        j += 1
plt.tight_layout()
plt.show()

# ----------------------------- #
# 4. Multiple linear regression
predictions = []
# select points by cluster
for u in np.unique(gdf.cluster):
    sel = gdf[gdf['cluster'] == u]
    if len(sel) < 100:
        continue
    
    # split to train test
    X = sel[br_cols]
    y = sel['field_depth']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123)
    
    # mlr
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    # predict
    pred = mlr.predict(X_test)
    score = mlr.score(X_test, y_test)
    
    # rmse
    mse = metrics.mean_squared_error(pred, y_test)
    rmse = np.sqrt(mse)
    
    # plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, s=0.5, color='Blue')
    ax.set_ylabel('Predicted depth (m)')
    ax.set_xlabel('Field depth (m)')
    ax.plot(0,1, lw=0.7, color='Red')
    ax.grid()
    plt.tight_layout()
    plt.show()
    print('R2: %.2f' % (score))
    print('RMSE: %.2f' % (rmse))
    
    
    # reshape band ratios for predict
    predict_stack = np.reshape(band_ratios, (band_ratios.shape[0], band_ratios.shape[1]*band_ratios.shape[2])).T
    # replace nan or -inf
    predict_stack = np.where(np.isfinite(predict_stack), predict_stack, 0)
    # predict
    prediction = mlr.predict(predict_stack)
    
    # reshape to 2D 
    predicted = prediction.reshape((band_ratios.shape[1], band_ratios.shape[2]))
    
    # mask nodata
    predicted = np.where(mask == True, np.nan, predicted)
    # drop negative values
    predicted = np.where(predicted < 0, np.nan, predicted)
  
    # mask outliers
    predicted = maskSigma3Outliers(predicted)
    # add to list
    predictions.append(predicted)    
    #plt.imshow(predicted)
    
# predictions to array
cbr = np.dstack(predictions)
# compute mean along 3rd axis
cbr_sdb = np.nanmean(cbr, axis=2)

plt.imshow(cbr_sdb)

outfile = os.path.join(os.path.dirname(fp), 'sdb', 'CBR_SDB.tif')
outmeta = meta.copy()
outmeta.update(count=1)

with rio.open(outfile, 'w', **outmeta) as dst:
    dst.write(cbr_sdb.astype(rio.float32),1)
























