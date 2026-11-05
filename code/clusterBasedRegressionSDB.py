# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:00:48 2023

Satellite derived bathymetry using cluster based regression
Geyman & Maloof 2019
https://doi.org/10.1029/2018EA000539

Steps
1. K-means clustering
2. Log band ratios (Stumpf et al.)

TODO: Handle points that are within same pixel
    Spatial autocorrelation in model accuracy assessment

@author: E1008409
"""

from rasterio.transform import rowcol, xy
import pandas as pd
import geopandas as gpd
import rasterio as rio
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#import statsmodels.api as sm
#import statsmodels.formula.api as smf

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

def sampleRaster(raster_fp, geodataframe_fp, colnames):
    # Read sampling points
    geodf = gpd.read_file(
        geodataframe_fp,
        engine="pyogrio"
    )

    with rio.open(raster_fp) as src:
        # Validate CRS
        if geodf.crs is None:
            raise ValueError("The input GeoDataFrame has no CRS.")

        if src.crs is None:
            raise ValueError("The raster has no CRS.")

        # Reproject points into the raster CRS
        if geodf.crs != src.crs:
            geodf = geodf.to_crs(src.crs)

        # Explode the entire GeoDataFrame, not only its geometry
        geodf = geodf.explode(
            index_parts=False,
            ignore_index=True
        )

        # Ensure all remaining geometries are Points
        invalid_types = geodf.geometry.geom_type.ne("Point")

        if invalid_types.any():
            types = geodf.loc[
                invalid_types,
                geodf.geometry.name
            ].geom_type.unique()

            raise TypeError(
                "All geometries must be Point or MultiPoint. "
                f"Found: {types.tolist()}"
            )

        # Extract x, y coordinates
        coords = list(zip(
            geodf.geometry.x,
            geodf.geometry.y
        ))

        # Sample one array per coordinate
        sampled = list(src.sample(coords, masked=True))

        if len(colnames) != src.count:
            raise ValueError(
                f"The raster has {src.count} bands, but "
                f"{len(colnames)} column names were provided."
            )
        
        # check sampled data type
        if isinstance(sampled[0], np.ma.MaskedArray):
            sampled_data = [s.data for s in sampled]
        else:
            sampled_data = sampled
        # Convert samples to a DataFrame with matching row index
        sample_df = pd.DataFrame(
            sampled_data,
            columns=colnames,
            index=geodf.index
        )

        # Assign sampled band values
        geodf[colnames] = sample_df

    return geodf

def sampleToGDF(raster_fp, geodataframe, colnames):

    with rio.open(raster_fp) as src:
        # Validate CRS
        if geodataframe.crs is None:
            raise ValueError("The input GeoDataFrame has no CRS.")

        if src.crs is None:
            raise ValueError("The raster has no CRS.")

        # Reproject points into the raster CRS
        if geodataframe.crs != src.crs:
            geodataframe = geodataframe.to_crs(src.crs)

        # Explode the entire GeoDataFrame, not only its geometry
        geodataframe = geodataframe.explode(
            index_parts=False,
            ignore_index=True
        )

        # Ensure all remaining geometries are Points
        invalid_types = geodataframe.geometry.geom_type.ne("Point")

        if invalid_types.any():
            types = geodataframe.loc[
                invalid_types,
                geodataframe.geometry.name
            ].geom_type.unique()

            raise TypeError(
                "All geometries must be Point or MultiPoint. "
                f"Found: {types.tolist()}"
            )

        # Extract x, y coordinates
        coords = list(zip(
            geodataframe.geometry.x,
            geodataframe.geometry.y
        ))

        # Sample one array per coordinate
        sampled = list(src.sample(coords, masked=True))

        if len(colnames) != src.count:
            raise ValueError(
                f"The raster has {src.count} bands, but "
                f"{len(colnames)} column names were provided."
            )

        # Convert samples to a DataFrame with matching row index
        sample_df = pd.DataFrame(
            sampled,
            columns=colnames,
            index=geodataframe.index
        )

        # Assign sampled band values
        geodataframe[colnames] = sample_df

    return geodataframe

def aggregate_to_s2_pixels(
    atl24: gpd.GeoDataFrame,
    s2_raster: str,
    depth_col: str = "depth"
) -> gpd.GeoDataFrame:
    """
    Aggregate ATL24 depths to Sentinel-2 pixels using median depth.

    Output geometry is the center of each Sentinel-2 pixel.

    Parameters
    ----------
    atl24 : GeoDataFrame
        ATL24 point observations.
    s2_raster : str
        Sentinel-2 raster defining the target pixel grid.
    depth_col : str
        Column containing bathymetric depth.

    Returns
    -------
    GeoDataFrame
        One point per Sentinel-2 pixel with:
        - depth: median depth
        - n_obs: number of ATL24 observations
        - geometry: Sentinel-2 pixel center
    """

    with rio.open(s2_raster) as src:

        # Reproject ATL24 points to Sentinel-2 CRS
        gdf = atl24.to_crs(src.crs).copy()

        # Remove missing depths
        gdf = gdf.dropna(subset=[depth_col])

        # Find Sentinel-2 pixel for every point
        rows, cols = rowcol(
            src.transform,
            gdf.geometry.x,
            gdf.geometry.y
        )

        gdf["row"] = rows
        gdf["col"] = cols

        # Keep only points inside the raster
        gdf = gdf[
            (gdf["row"] >= 0) &
            (gdf["row"] < src.height) &
            (gdf["col"] >= 0) &
            (gdf["col"] < src.width)
        ]

        # Median depth per pixel
        pixels = (
            gdf.groupby(["row", "col"])
            .agg(
                depth=(depth_col, "median"),
                n_obs=(depth_col, "size")
            )
            .reset_index()
        )

        # Exact Sentinel-2 pixel-center coordinates
        x, y = xy(
            src.transform,
            pixels["row"],
            pixels["col"],
            offset="center"
        )

        pixels["geometry"] = gpd.points_from_xy(x, y)

        return gpd.GeoDataFrame(
            pixels,
            geometry="geometry",
            crs=src.crs
        )

# filepaths
fp = "/mnt/d/BlueZan/S2/downloads/Chwaka-NESAP_S2B_20260723T072619_10m_bands.tif"
# ground truth
fp_gt = "/mnt/d/BlueZan/Field_data/BlueZan_habitat_data_11_2024_labels_edit.gpkg"
fp_atl24 = '/mnt/d/BlueZan/ICESAT-2/atl24_conf80_depths_chwaka-nesap.gpkg'
fp_kmeans = "/mnt/d/BlueZan/S2/annotation/classification_2026/randomForest26_classification_s2_2026.tif"

USE_CLASSIFICATION_MAP = True
DEPTH_COL = 'depth'

# ------------------------ # 
# Read ATL24 data and filter 1 obs per pixel
# ------------------------ #
atl24 = gpd.read_file(fp_atl24)

depth_pixels = aggregate_to_s2_pixels(
    atl24, fp, depth_col="depth_m")
# save
filtered_points = fp_atl24.split('.')[0] + '_filtered.gpkg'
depth_pixels.to_file(
    filtered_points,
    driver="GPKG"
)

# ------------------------ # 
# 1. K-means unsupervised classification
# ------------------------ #
if USE_CLASSIFICATION_MAP:
    # read KMeans clustered image
    with rio.open(fp_kmeans) as src:
        kmeans = src.read()
        kmeans_prof = src.profile
else:
    # read KMeans clustered image
    with rio.open(fp_kmeans) as src:
        kmeans = src.read()
        kmeans_prof = src.profile
    # update nodata
    kmeans_prof.update(nodata=0)
    # save
    kmeans_out = fp_kmeans.split('.')[0] + '_nodata_updated.tif'
    with rio.open(kmeans_out, 'w', **kmeans_prof) as dst:
        dst.write(kmeans)

# ----------------------------- # 
# 2. Read field data and prepare dataset
# sample
if USE_CLASSIFICATION_MAP:
    gdf = sampleRaster(fp_kmeans, filtered_points, ['kmeans'])
else:    
    gdf = sampleRaster(kmeans_out, filtered_points, ['kmeans'])

# fix depths to floats
if gdf[DEPTH_COL].dtype != 'float64':
    # convert syvyys_mitattu to numeric
    gdf[DEPTH_COL] = gdf[DEPTH_COL].astype(float)
else:
    # absolute values
    gdf[DEPTH_COL] = abs(gdf[DEPTH_COL])

# --------------------------- #
# 2. Log band ratios
with rio.open(fp) as src:
    img = src.read()
    profile = src.profile
# list of unique band pairs
unique_pairs = list(itertools.combinations(np.arange(1,4), 2))
# compute band ratios
band_ratios = [StumpfLogBandRatio(img, u) for u in unique_pairs]
# to array
band_ratios = np.array(band_ratios)
# save stack for sampling
outdir = os.path.join(os.path.dirname(fp), 'sdb')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
stack_out = os.path.join(outdir, 'band_ratios_temp.tif')

stack_meta = profile.copy()
stack_meta.update(count=band_ratios.shape[0],
                  nodata=np.nan,
                  dtype='float32')

with rio.open(stack_out, 'w', **stack_meta) as dst:
    dst.write(band_ratios)

# sample band ratios to points
logbr_cols = [f'br_{b}' for b in np.arange(1, len(band_ratios)+1) ]
gdf = sampleToGDF(stack_out, gdf, logbr_cols)

# ----------------------------- #
# 4. Multiple linear regression
predictions = np.zeros(shape=kmeans.shape)
# select points by cluster
for u in np.unique(gdf['kmeans']):
    sel = gdf[gdf['kmeans'] == u]
    if len(sel) < 100:
        continue
    
    # set data 
    X = sel[logbr_cols]
    y = sel[DEPTH_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123)
    # reg
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    # predict
    pred = reg.predict(X_test)
    score = reg.score(X_test, y_test)
    
    # rmse
    mse = metrics.mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    # mae
    mae = metrics.mean_absolute_error(y_test, pred)
    
    # plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, pred, s=0.5, color='Blue')
    ax.set_ylabel('Predicted depth (m)')
    ax.set_xlabel('Field depth (m)')
    ax.plot(0,1, lw=0.7, color='Red')
    ax.grid()
    ax.text(0.1, max(pred)-0.1, 'RMSE: %.2f' % (rmse))
    ax.text(0.1, max(pred)-1, 'R2: %.2f' % (score))
    plt.tight_layout()
    plt.show()
    print('R2: %.2f' % (score))
    print('RMSE: %.2f' % (rmse))
    
    # create mask
    mask = np.where(kmeans == u, True, False)
    # reshape band ratios for predict
    predict_stack = np.reshape(band_ratios, (band_ratios.shape[0], band_ratios.shape[1]*band_ratios.shape[2])).T
    # replace nan or -inf
    predict_stack = np.where(np.isfinite(predict_stack), predict_stack, 0)
    # predict
    prediction = reg.predict(predict_stack)
        # reshape to 2D 
    predicted = prediction.reshape((band_ratios.shape[1], band_ratios.shape[2]))
    
    # mask nodata
    predictions = np.where(mask == True, predicted, predictions)
    
# predictions to array
#cbr = np.dstack(predictions)
# compute mean along 3rd axis
#cbr_sdb = np.nanmean(cbr, axis=2)


outfile = os.path.join(os.path.dirname(fp), 'sdb', 'CBR_SDB.tif')
outmeta = profile.copy()
outmeta.update(count=1,
               dtype='float32',
               nodata=0)

with rio.open(outfile, 'w', **outmeta) as dst:
    dst.write(predictions)

# --------------------------- #
# plot predicted depths against field measurements
gt = sampleRaster(outfile, fp_gt, ['SDB'])

# fix depths to floats
if gt['Depth'].dtype != 'float64':
    # convert syvyys_mitattu to numeric
    gt['Depth'] = gt['Depth'].astype(float)
else:
    # absolute values
    gt['Depth'] = abs(gt['Depth'])
# ensure all sampled are values, drop row if not
gt = gt[gt['SDB'] != '--']

# rmse
mse = metrics.mean_squared_error(gt['Depth'], gt['SDB'])
rmse = np.sqrt(mse)
# mae
mae = metrics.mean_absolute_error(y_test, pred)

# plot
fig, ax = plt.subplots()
ax.scatter(gt['Depth'], gt['SDB'], s=0.5, color='Blue')
ax.set_ylabel('Predicted depth (m)')
ax.set_xlabel('Field depth (m)')
ax.plot(0,1, lw=0.7, color='Red')
ax.grid()
ax.text(0.1, max(gt['SDB'])-0.1, 'RMSE: %.2f' % (rmse))
ax.text(0.1, max(gt['SDB'])-1, 'R2: %.2f' % (score))
plt.tight_layout()
plt.show()
print('R2: %.2f' % (score))
print('RMSE: %.2f' % (rmse))




















