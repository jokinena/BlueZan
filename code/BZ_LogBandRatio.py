# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 12:24:48 2026

@author: Matkauser
"""

import os
import numpy as np
import rasterio as rio

# function to compute logarithmic band ratio from given bands
def StumpfLogBandRatio(image_fp, bands):
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
    with rio.open(image_fp) as src:
        image_array = src.read(masked=True)
        profile = src.profile
    
    # logbr 
    logbr = (np.log(1000*image_array[bands[0]])) / (np.log(1000*image_array[bands[1]]))
    logbr = logbr.reshape((logbr.shape[0], logbr.shape[1]))
    
    # mask nodata
    mask2d = np.any(image_array.mask, axis=0)   # True where any input channel is masked    
    logbr = np.ma.where(mask2d, np.nan, logbr)
    
    # save
    outname = os.path.join(os.path.dirname(image_fp), f'logbr_bands_{bands[0]}_{bands[1]}.tif')
    outprof = profile.copy()
    outprof.update(dtype='float32',
                   count=1,
                   nodata=np.nan)
    with rio.open(outname, 'w', **outprof) as dst:
        dst.write(logbr,1)
    
    
image_fp = '/mnt/d/BlueZan/S2/downloads/Chwaka-NESAP_S2B_20260723T072619_10m_bands.tif'
bands = (0, 1) # bands where to compute log-ratio

if __name__ == '__main__':
    StumpfLogBandRatio(image_fp, bands)
    
    
    
    
    
    
    
    
    