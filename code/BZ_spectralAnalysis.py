# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 19:27:25 2026

BlueZan 


@author: E1008409
"""

import os
import time 
import numpy as np
import geopandas as gpd
import rasterio as rio
from sklearn.cluster import KMeans
from skimage.segmentation import felzenszwalb
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import permanova,  DistanceMatrix
from itertools import combinations
from dask import delayed, compute

def sampleRaster(raster_fp, geodataframe_fp, colname_prefix):
    # read points
    gdf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        #desc = src.descriptions # optional TODO: check if band names have descriptions and use those to in colname extraction
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
        # create names for sampled columns
        colnames = [colname_prefix + str(i) for i in np.arange(1,len(gdf.sampled[0])+1,1)]
        # extract list
        gdf[colnames] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        # drop sampled
        gdf = gdf.drop('sampled', axis=1)
    return gdf

def KMeansClusters(raster_fp, n_clusters):
    # read raster
    with rio.open(raster_fp) as src:
        img = src.read()
        profile = src.profile
        # reshape to 2D array (pixels, bands)
        img_reshaped = img.reshape(img.shape[0], -1).T  # shape: (num_pixels, num_bands)
        
    # perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img_reshaped)
    
    # get cluster labels and reshape back to original image shape
    labels = kmeans.labels_.reshape(img.shape[1], img.shape[2])
    
    return labels, profile

def saveRaster(data, profile, output_fp):
    with rio.open(output_fp, 'w', **profile) as dst:
        dst.write(data)

def segmentImage(raster_fp, n, sig, s):
    # read raster
    with rio.open(raster_fp) as src:
        img = src.read()
        profile = src.profile
    print('Image shape:', img.shape)
    
    # felzenswalb segmentation 
    start = time.time()
    segments = felzenszwalb(img[0:3], scale=s, sigma=sig, min_size=n, channel_axis=0) 

    end = time.time()
    elapsed = end - start
    print('1st segmentation, Time elapsed: %.2f' % elapsed, 'seconds')
            
    # mask nodata areas
    nodatamask = np.where((img[2] == profile['nodata']) | (np.isnan(img[2])), True, False)
    segments = np.where(nodatamask == True, 0, segments)
    # check that all segments are not nodata
    if not np.any(segments) == True:
        print('All nodata')
        
    # expand dims
    segments = np.expand_dims(segments, axis=0)
            
    # define profile
    segmeta = profile.copy()
    segmeta.update(dtype='int32',
                    width=segments.shape[2],
                    height=segments.shape[1],
                    nodata=0,
                    count=1,
                    compress='lzw')
    segdir = os.path.dirname(raster_fp) + '/segmentation'
    if os.path.isdir(segdir) == False:
        os.mkdir(segdir)
    
    segments_out = os.path.join(segdir, 'segments_' + '_n' + str(n) + '_s' + '.tif') 
    print('Saving segments')

    # save
    with rio.open(segments_out, 'w', **segmeta) as dst:
        dst.write(segments.astype(segmeta['dtype']))

    return segments, segments_out

# filepaths
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BlueZan_habitat_data_11_2024_labels.gpkg'
fp_img = '/mnt/d/users/e1008409/MK/sansibar/S2/Chwaka/S2A_MSIL2A_20240429_B2B3B4B8_masked_chwaka.tif'

# read data
with rio.open(fp_img) as src:
    img = src.read()
    profile = src.profile
# cluster image
n = 10
clusters, kmeans_profile = KMeansClusters(fp_img, n_clusters=n)
# save
kmeans_profile.update(count=1, dtype='int32')
kmeans_out = os.path.join(os.path.dirname(fp_img), f'kmeans_{n}clusters.tif')
saveRaster(clusters[np.newaxis, :, :], kmeans_profile, kmeans_out)

# image segmentation
segments, segments_out = segmentImage(fp_img, n=10, sig=0.1, s=0.50)
# join habitat from points to clusters of image pixels and look for habitat spectral similarity
gdf = sampleRaster(segments_out, fp_pts, colname_prefix='segment_')
# save 
gdf_out = fp_pts.split('.')[0] + '_segments.gpkg'
gdf.to_file(gdf_out)

def getSegmentPixels(image, segment):
    # select pixels for point
    mask = segments==segment
    squeezed_mask = mask.squeeze(0)
    pix = img[:, squeezed_mask]
    
    return pix

# dict for segment pixel values
pixvals = {}
    
for seg_id in gdf['segment_1'].unique():
    
    pix = getSegmentPixels(img, seg_id)
    # to dict
    pixvals[seg_id] = {'pixvals': pix,
                       'hab': gdf['subs_highest'].loc[gdf.index[gdf['segment_1']==seg_id]].values[0]}


# compare spectral separability
combs = [c for c in combinations(gdf['segment_1'], r=2)]

def permanova(c1, c2):
    # 1. Define pixel groups
    group1 = pixvals[c1] 
    group2 = pixvals[c2]
    
    # 2. Transpose the arrays so pixels are rows, and bands are columns (.T)
    X_group1 = group1['pixvals'].T  # Shape becomes (38, 4)
    X_group2 = group2['pixvals'].T  # Shape becomes (3, 4)
    
    # 3. Combine the pixel data and create matching labels
    X_combined = np.vstack([X_group1, X_group2]) 
    
    labels = ([group1['hab']] * X_group1.shape[0]) + ([group2['hab']] * X_group2.shape[0])
    
    # 4. Compute the pairwise Euclidean Distance Matrix
    # pdist calculates condensed distances, squareform converts it to a square matrix
    distance_matrix_condensed = pdist(X_combined, metric='euclidean')
    distance_matrix_square = squareform(distance_matrix_condensed)
    # Convert to DistanceMatrix
    skbio_distance_matrix = DistanceMatrix(distance_matrix_square)
    
    # Run PERMANOVA
    permanova_result = permanova(skbio_distance_matrix, labels, permutations=999)
    
    return permanova_result
    
def spectralSeparability(c1, c2):
    # 1. Define pixel groups
    group1 = pixvals[c1] 
    group2 = pixvals[c2]
    # Mann-Whitney U-test
    mwu_stat, mwu_p = stats.mannwhitneyu(group1['pixvals'].T, group2['pixvals'].T, alternative='two-sided', axis=0, method='exact')
    # update dict
    pixvals[c1][f'against_{c2}'] = {'U-stat': mwu_stat,
                                   'p-val': mwu_p,
                                   'to_segment': c2}
    # check if non-significant, ie. not separable
    if any(mwu_p > 0.05):
        indcs = np.argwhere(mwu_p > 0.05)
        # update dict
        pixvals[c1][f'against_{c2}'].update({'nonsignificant': True,
                                           'non_s_bands': indcs})
    # check if significant between same habitat
    hab1 = pixvals[c1]['hab']
    hab2 = pixvals[c2]['hab']
    if hab1 == hab2:    
        pixvals[c1][f'against_{c2}'].update({'same_habitat': True})
    
# delayed funcs
delayed_funcs = [] 
for c in combs:
    y = delayed(spectralSeparability)(c[0], c[1])
    delayed_funcs.append(y)
# compute
compute(*delayed_funcs)

# find segments per habitat type which are different



# points with homogenous habitat, eg. 100% seagrass, or sand. Compare reflectance to mixed points at similar depth interval

# if classifier is uncertain and pixel is in a spectrally separable cluster, classify as unidentified habitat?









































