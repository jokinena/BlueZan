# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 08:38:04 2025

Dark Object Subtraction (DOS) is an empirical atmospheric correction method which assumes that
reflectance from dark objects includes a substantial component of atmospheric scattering. This
atmospheric offset generated from the image itself can be removed by subtracting this value from
every pixel in the band. However, this value is different for each band and can be also estimated as
the value of the histogram's cut-off point at the lower end (Chavez, 1998). The most effective dark
target would be optically-deep water with expected zero reflectance. 

@author: E1008409
"""

import argparse
import glob
import os
import numpy as np
import rasterio as rio
import matplotlib.pyplot as plt


CLI = argparse.ArgumentParser()
CLI.add_argument('filepath',
                 type=str,
                 help='Filepath for field data points')
args = CLI.parse_args()
filepath = args.filepath

fp = '/mnt/d/users/e1008409/MK/sansibar/S2/S2C_MSIL1C_20250524T072631_N0511_R049_T37MEP_20250524T092906_B02B03B04B08.tif'

def DOS(fp):
    # read
    with rio.open(fp) as src:
        img = src.read()
        profile = src.profile
    nodatamask = np.where(img == 0, True, False)
    band_list = []
    # visible light bands each in turn
    for i in range(0,len(img)):
        print(i)
        band = img[i]
        # compute histogram
        hist, bin_edges = np.histogram(img[i], np.arange(0, np.nanmax(img[i]), 1))
        # get index of first nonzero value
        indcs = np.argwhere(hist).squeeze() # all nonzero indices
        ind = indcs[1]
        # get frequency at index location
        freq = hist[ind]
        # find threshold where freq starts rising
        for j in hist[1:]:
            if j >= freq*3: # TODO: this is arbitrary, define better later
                print(i)
                i_loc = np.argwhere(hist == j)[0]
                # get bin edge value at i_loc
                bin_value = bin_edges[i_loc][0]
                break
        # get bin edge value
        #bin_ind = bin_edges[ind[1]]
        
        # plot
        fig, ax = plt.subplots()
        ax.stairs(hist[ind:ind+20], bin_edges[ind:ind+21], fill=True) # ,
#        ax.vlines(bin_edges[ind[1:]], bin_edges[ind[1:]], hist.max(), colors='w')
        ax.axvline(bin_value, ls='--', lw=0.5, color='black')
        ax.set_xlabel('DN')
        ax.set_ylabel('Frequency')
        ax.set_title('DOS threshold for band ' + str(i+1))        
        plt.show()
        
        # get min value > 0 from bin edges
        minvalue = bin_value
        print('Min value', minvalue)
        # subtract minvalue from band
        dos_band = band - minvalue
        # mask possible negative values
        dos_band = np.where(dos_band < 0, 0, dos_band)
        band_list.append(dos_band)   
    
    # stack dos bands
    dos = np.stack(band_list)
    # mask nodata area
    dos = np.where(nodatamask == True, 0, dos)
    # output name
    outname = fp.split('.tif')[0] + '_dos.tif'
    
    # write out multiband tif file from band list
    with rio.open(outname, 'w', **profile) as dst:
        dst.write(dos)

def main():
    DOS(fp)

if __name__ == '__main__':
    main()









