# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:49:32 2026

Plot estimated habitat area from probability raster

@author: A-P
"""


import os 
from pathlib import Path
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt

# dirpath for pred rasters
pred_dir = Path('/mnt/d/BlueZan/S2/annotation/masked_proba')

# set threshold and years
THRESHOLD = 0.7
years = [2018, 2022, 2026]
areas = []

# read image
def computeArea(image_path, threshold):
    # read image
    with rio.open(image_path) as src:
        img = src.read()
        profile = src.profile
    # select by threshold
    pixels = img[img >= threshold]
    # count area
    pixel_size = profile['transform'][0]
    area_ha = np.shape(pixels)[0] * pixel_size * pixel_size / 10000
    
    return area_ha

if __name__ == '__main__':
    # list files
    files = [pred_dir/f for f in os.listdir(pred_dir)]
    
    # compute areas per image
    for f in files:
        area = computeArea(f, THRESHOLD)
        areas.append(area)
    
    # plot areas
    fig, ax = plt.subplots()
    ax.plot(years, areas)
    ax.set_xticks(years)
    ax.set_ylabel('Area (ha)')
    ax.grid()
    ax.set_title('Seagrass area estimate from classification maps')
    
    plot_out = pred_dir/'area_estimate_plot.png'
    plt.savefig(plot_out, dpi=150)
    #plt.show()




