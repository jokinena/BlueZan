# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 13:56:50 2025

Create feature space plot

@author: E1008409
"""


import os 
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from collections import Counter
import seaborn as sns
from scipy import stats

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
    return gdf

# -----------------------------------------------------
# files
fp = '/mnt/d/users/e1008409/MK/sansibar/planet/ChwakaBay_04-2024_psscene_analytic_8b_sr_udm2/20240430_3B_AnalyticMS_SR_8b_merge.tif'
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BZC20241107_1st_timestamp.gpkg'


# define output folder for plots
outdir = os.path.join(os.path.dirname(fp_pts), 'plots')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
prefix = os.path.basename(fp_pts).split('_')[0]
# sample raster
gdf = sampleRaster(fp, fp_pts)
print(gdf.columns)
# create names for sampled columns
colnames = ['Band' + str(i) for i in np.arange(1,len(gdf.sampled[0])+1,1)]
# extract list
gdf[colnames] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
gdf = gdf.drop('sampled', axis=1)

# save gdf
gdfout = fp_pts.split('.gpkg')[0] + '_sampled.gpkg'
gdf.to_file(gdfout)

# remove outliers
bands = gdf.columns[-8:].tolist()
print(bands)
for b in bands:
    gdf = gdf[np.abs(stats.zscore(gdf[b])) < 3]
print(np.unique(gdf.pseudoclass, return_counts=True))
# set keys and colors
keys = sorted(gdf.pseudoclass.unique().tolist())
print(keys)

colors = ['green', 'orange', 'purple', 'blue'] # 

# create dict
palette = {keys[i]: colors[i] for i in range(len(keys))}

sel = gdf[['Band2', 'Band3', 'Band5', 'Band6','pseudoclass']] # use Band8 for NL as it is tidal
figout = os.path.join(outdir, 'FSpace_data_SNS.png')
g = sns.pairplot(sel, hue='pseudoclass', palette=palette, plot_kws={'s':3})
g.fig.suptitle('Planet SuperDove feature space', y=1.08) # with Kernel density estimation
g._legend.set_title('Classes')
g.savefig(figout, dpi=300)

















