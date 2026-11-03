# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 15:35:54 2021

Apply sun glint correction Hedley et al. (2005) to satellite image. 
Additionally, uses NDVI to mask non-water pixels

One or more regions of the image are selected (polygons) where a range of sun glint is evident, 
but where the underlying spectral brightness would be expected to be consistent (areas of deep water are ideal for this)

Sun glint correction R'i = Ri - bi *(Rnir - minNIR) Hedley et al. (2005)
    R'i - deglinted pixel
    Ri - reflectance from visible band i
    bi - regression slope
    Rnir - NIR band value
    minNIR - minimum NIR value in the sample
    
@author: Ari-Pekka Jokinen
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import glob
import os
import rasterio as rio
from rasterio.mask import mask
import matplotlib.pyplot as plt


def normalizedDifference(image_array, band_indices):
    """
    Compute normalized difference from given bands

    Parameters
    ----------
    image_array : array
        DESCRIPTION.
    band_indices : integer tuple
        DESCRIPTION.

    Returns
    -------
    Normalized difference

    """
    # computation
    ndiff = (image_array[band_indices[0]].astype(float) - image_array[band_indices[1]].astype(float)) / (image_array[band_indices[0]] + image_array[band_indices[1]])
    return ndiff
  
# sansibar
fp = '/mnt/d/BlueZan/S2/downloads/Chwaka-NESAP_S2B_20260723T072619_10m_bands.tif' # filepath for satellite image

fp_poly = '/mnt/d/BlueZan/shapes/Chwaka-NESAP_deglint.gpkg'

outdir = os.path.join(os.path.dirname(fp), 'deglint')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)

if __name__ == '__main__':

    # read polygon
    gdf = gpd.read_file(fp_poly, engine='pyogrio')
    geoms = gdf.geometry.values

    # set nir_band (starting at 1)
    nir_band = 4
    nd_mask = False

    # perform sun glint correction for each file
    for file in glob.glob(fp):
        #if '3c40' in file: # option to select certain file
        print('Working on file: ' + os.path.basename(file))
        with rio.open(file) as src:
           img = src.read()
           meta = src.meta
           nodata = src.nodata
           # extract pixel values inside the polygons
           out_image, out_transform = mask(src, geoms, crop=True)

            # if only zeros in masked array print message and continue
        if np.all(out_image == 0):
            print('Only nodata in masked arrays, continuing to next image')
            continue

        # nodata mask
        nodata_mask = np.where(img[0] == nodata, True, False)
        if nd_mask == True:
            # NDWI
            ndvi = normalizedDifference(img, (2,7))    
            # mask non-water area
            ndvi_landmask = np.where(ndvi >= 0, True, False)
            img_ndviland = np.where(ndvi_landmask == True, img, nodata)    
            img = np.where(ndvi < 0, img, nodata)

        # image specific: scale image by 10000
        img = img / 10000
        out_image = out_image / 10000
        band_list = []
        
        b_nir = out_image[nir_band-1] # get the NIR band data 
        # mask nodata
        b_nir = np.where(b_nir == nodata, np.nan, b_nir) # replace nodata with np.nan
        b_nir_mask = ~np.isnan(b_nir) 
        bnir = b_nir[b_nir_mask] #keep valid values
        bnir_re = bnir.reshape(-1,1)
        # visible light bands each in turn
        for i in range(0,len(img)):
            if i+1 >= nir_band: # keep nir band as is
                full_band = img[i]
                nir = np.where(nodata_mask == True, nodata, full_band)
                minNIR = np.nanmin(bnir_re) # minimum NIR value
                nir = nir - minNIR # subtract 
                band_list.append(nir)
            else:
                print('Processing band: ', i+1)
                full_band = img[i]
                band_data = out_image[i]
    #            band_data = np.where(band_data == nodata, np.nan, band_data)
    #           band_mask = ~np.isnan(band_data)
    #            band_data_m = band_data[band_mask]
                # mask 
                band_data_m = band_data[b_nir_mask]
                band_data_re = band_data_m.reshape(1,-1)
                
                # linear regression of the band values
                model = LinearRegression().fit(bnir_re, band_data_re[0])
                minNIR = np.nanmin(bnir_re) # minimum NIR value from the sample
                intercept = model.intercept_
                slope = model.coef_[0]
                # print scatter plot 
                r_sq = 'R^2: ' + str(round(model.score(bnir_re, band_data_re[0]), 3))
                figname = os.path.join(outdir, 'deglint_band' + str(i+1) + '.png')
                fig, ax = plt.subplots()
                ax.scatter(bnir_re, band_data_re, s=0.5, c='k')
                ax.text(min(bnir_re), max(band_data_re[0]), r_sq)
                ax.plot(bnir_re, slope*bnir_re + intercept)
                ax.set_xlabel('NIR')
                ax.set_ylabel('Band ' + str(i+1))
                ax.set_title(os.path.join('Band ' + str(i+1) + ' ' + os.path.basename(file)))
    #            plt.savefig(figname, dpi=150)
                plt.show()
                # deglint  Hedley et al. (2005)
                band_deglint = full_band - slope * (img[nir_band-1] - minNIR)       
                # mask nodata area
                deglinted_band = np.where(nodata_mask == True, nodata, band_deglint)
                # mask negative values
                deglinted_band = np.where(deglinted_band < 0, nodata, deglinted_band)
                band_list.append(deglinted_band)   
        
        # stack deglinted bands
        deglinted = np.stack(band_list)
        # add no-water area to deglinted image
        #deglinted = np.where(ndvi_landmask == True, img_ndviland, deglinted)
        # write out
        print('Writing file')
        # output name
        bname = os.path.basename(file)[:-4]
        outname = os.path.join(outdir, bname + '_deglint.tif')
        
        # update metadata
        meta.update(nodata=0,
                    dtype=np.float32)
        
        # write out multiband tif file from band list
        with rio.open(outname, 'w', **meta, compress='LZW') as dst:
    #        for band_nr, band_array in enumerate(band_list, start=1):            
            dst.write(deglinted.astype(meta['dtype']))























