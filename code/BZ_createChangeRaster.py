# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 16:50:20 2026

Change between two prediction rasters

@author: E1008409
"""


import os 
import geopandas as gpd
import rasterio as rio
import numpy as np


fp1 = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/classification/randomForest_classification_s2.tif'
fp2 = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/classification_2025/randomForest25_classification_s2_2025.tif'
fp_annt = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/Chwaka_annotation.gpkg'
label = 'Seagrass meadow'

def readRaster(fp):
    # read 
    with rio.open(fp) as src:
        img = src.read()
        profile = src.profile
    
    return img, profile

def getLabels(fp_annt):
    gdf = gpd.read_file(fp_annt)
    labels = sorted(gdf.label.unique())

    return labels

def getLabelIndex(labels, label):
    l_ind = labels.index(label)
    
    return l_ind

def change(image1, image2, fp_annt, label):
    # read images    
    img1, p1 = readRaster(fp1)
    img2, p2 = readRaster(fp2)
    # get labels
    labels = getLabels(fp_annt)
    # get label index
    l_ind = getLabelIndex(labels, label)
    # select class by index. Exclude 0 as nodata
    cl = np.unique(img1)[1:][l_ind]
    
    # create change raster for selected class    
    change = np.where((img1 == cl) & (img2 == cl), 1, 0)
    change = np.where((img1 == cl) & (img2 != cl), 2, change)
    change = np.where((img1 != cl) & (img2 == cl), 3, change)
    # mask nodata areas
    change = np.where(img1 == p1['nodata'], 0, change)
    change = np.where(img2 == p2['nodata'], 0, change)
    
    # check if spaces in label name
    test = label.split(" ")
    if len(test) > 1:
        label = label.replace(" ", "_")
    
    # save change raster
    outfile = os.path.join(os.path.dirname(fp_annt), label + '_change.tif')
    with rio.open(outfile, 'w', **p1) as dst:
        dst.write(change)
    
def main():
    change(fp1, fp2, fp_annt, label)

if __name__ == '__main__':
    main()














