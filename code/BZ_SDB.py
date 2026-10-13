# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 13:41:13 2025

Create satellite derived bathymetry based on Stumpf et al. (2003). Determination of water depth with high‐resolution satellite imagery over variable bottom types. Limnology and Oceanography, 48(1part2), Art. 1part2.


@author: E1008409
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import math
import gc

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# function to compute logarithmic band ratio from given bands
def StumpfLogBandRatio(image_array, bands):
    """
    Compute logarithmic band ratio as ln(1000*pi*band) / ln(1000*pi*band), see Stumpf et al. (2003)

    Parameters
    ----------
    image_array : np.array
        Image array
    bands : tuple or list
        Band indices 

    Returns
    -------
    logbr : np.array
        Log-ratio of image bands

    """
    logbr = (np.log(1000*image_array[bands[0]])) / (np.log(1000*image_array[bands[1]]))
    logbr = logbr.reshape((logbr.shape[0], logbr.shape[1]))
    
    return logbr

def sampleRaster(raster_fp, geodataframe_fp):
    # read points
    gdf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        # check geometry, explode if MultiPoint
        if gdf.geometry.geom_type.str.contains('MultiPoint').any() == True:
            sp = gdf.geometry.explode()
            coords = [(x,y) for x,y in zip(sp.geometry.x, sp.geometry.y)]
        else:
            # get point coords
            coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
        # sample
        gdf['sampled'] = [x for x in src.sample(coords)]
        # extract values
        gdf['sampled'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
    return gdf


fp_img = '/mnt/d/users/E1008409/MK/sansibar/S2/deglint/S2A_MSIL2A_20240429_B2B3B4B8_masked_chwaka_deglint.tif'
#fp_pts = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/habitat_field_data.shp'
fp_pts = '/mnt/d/users/E1008409/MK/sansibar/Mission_10_2024/Field_work/BlueZan_habitat_data_11_2024_labels.gpkg'
#fp_pts = '/mnt/d/users/E1008409/MK/sansibar/Lyzenga/Waypoints_Lyzenga.shp'

outdir = os.path.join(os.path.dirname(fp_img), 'SDB')
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# read 
with rio.open(fp_img) as src:
    img = src.read()
    profile = src.profile
img = np.ma.masked_equal(img,0)
# pts
gdf = gpd.read_file(fp_pts, engine='pyogrio')
print(gdf.columns)

# compute log-ratio band
logbr = StumpfLogBandRatio(img, (2,0))
# save
outname = os.path.join(outdir, 'logbr.tif')
outprof = profile.copy()
outprof.update(count=1)
with rio.open(outname, 'w', **outprof) as dst:
    dst.write(logbr,1)

# sample log-band ratio to point observations
gdf = sampleRaster(outname, fp_pts)
# rename sampled column
gdf = gdf.rename(columns={'sampled': 'logbr'})
print(gdf.columns)
gdf['Depth'] = gdf['Depth'].astype(float)
# select sand bottom
#gdf = gdf[(gdf.Sand == 1) & (gdf.Seagrass == 0) & (gdf.Coral == 0) & (gdf.Mixed == 0) & (gdf.Macroalga == 0)]
# keep data up to 15m depth and more than 1
gdf = gdf[gdf['Depth'] < 18]
#gdf = gdf[gdf['Depth'] >= 2]


X = np.array(gdf.logbr).reshape(-1,1)
y = np.array(gdf['Depth'])

# fit model
reg = LinearRegression().fit(X, y)
# model
m0 = reg.intercept_
m1 = reg.coef_[0]
# plot
fig, ax = plt.subplots()
ax.scatter(X, y, s=0.5, color='Blue')
ax.axline(xy1=(0, m0), slope=m1, label=f'$y = {m1:.1f}x {m0:+.1f}$')
ax.set_xlim(np.min(X), np.max(X))
ax.set_ylim(np.min(y), np.max(y))
plt.legend()
plt.show()


# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123)

# fit
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)
# predict
pred = reg.predict(X_test)
score = reg.score(X_test, y_test)
# rmse
mse = metrics.mean_squared_error(pred, y_test)
rmse = np.sqrt(mse)
# model
m0 = reg.intercept_
m1 = reg.coef_[0]
# plot
fig, ax = plt.subplots(1,2)

ax[0].scatter(X_test, y_test, s=0.5, color='Blue')
ax[0].set_xlabel('Log band ratio')
ax[0].set_ylabel('Field depth (m)')
mn=np.min(X_test)
mx=np.max(X_test)
ax[0].axline((mn,mx), slope=m1, lw=0.5, ls='--', color='black', alpha=0.8, label=f'$y = {m1:.1f}x {m0:+.1f}$')
ax[0].grid()
ax[0].annotate('$R^2$: %.2f' % (score), (0.05, 0.9), xycoords='axes fraction')
ax[0].set_title('Linear regression')
ax[0].legend(loc='lower right')

ax[1].scatter(pred, y_test, s=0.5, color='Blue')
ax[1].set_xlabel('Predicted depth (m)')
ax[1].set_ylabel('Field depth (m)')
ax[1].axline((0,0), slope=1, lw=0.5, ls='--', color='black', alpha=0.8)
ax[1].grid()
ax[1].annotate('RMSE: %.2f' % (rmse), (0.05, 0.9), xycoords='axes fraction')
ax[1].set_title('Accuracy')
plt.tight_layout()
plt.show()

reg = LinearRegression().fit(X, y)


# reshape logbr for predict
logbr_re = np.reshape(logbr, (-1,1))
# replace nan
logbr_re = np.where(np.isnan(logbr_re), 0, logbr_re)
# predict
prediction = reg.predict(logbr_re)
print(np.nanmin(prediction))
print(np.nanmax(prediction))
# reshape to 2D 
predicted = prediction.reshape((1,logbr.shape[0], logbr.shape[1]))
# mask nodata
predicted = np.where(np.isnan(logbr), np.nan, predicted)

# save
bathy_out = os.path.join(outdir, 'logbr_bathy.tif')
with rio.open(bathy_out, 'w', **outprof) as dst:
    dst.write(predicted)




    
bathy = m1*logbr+m0
print(np.nanmin(bathy))
print(np.nanmax(bathy))
bathy = np.where(bathy < 0, np.nan, bathy)
# save
bathy_out = os.path.join(outdir, 'logbr_bathy.tif')
with rio.open(bathy_out, 'w', **outprof) as dst:
    dst.write(bathy,1)

















