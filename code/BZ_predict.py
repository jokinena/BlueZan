# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 07:15:47 2025

@author: E1008409
"""

import os 
import geopandas as gpd
import rasterio as rio
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import geopandas as gpd

# filepaths
fp_img = '/mnt/d/BlueZan/S2/downloads/deglint/Chwaka-NESAP_S2B_20260723T072619_10m_bands_deglint.tif' # filepath for satellite image
fp_model = '/mnt/d/BlueZan/S2/annotation/classification_2026/randomForest26.sav' # full filepath to model
fp = '/mnt/d/BlueZan/S2/annotation/Chwaka_annotation.gpkg' # filepath for field points

# Set True if predict probabilities
PREDICT_PROBA = True
SAVE_SINGLE_BAND_PROBA = True
# set descriptions from labels
gdf = gpd.read_file(fp)
descriptions = [f'{n}_proba' for n in np.unique(gdf.label)]

# No need to edit below
# -------------------------------------------------------------- #
modelname = os.path.basename(fp_model).split('.sav')[0]

def readModel(fp_model):
    # read model 
    clf = pickle.load(open(fp_model, 'rb'))
    outdir = os.path.dirname(fp_model)
    return clf, outdir

def readImage(fp_image):
    # read image
    with rio.open(fp_img) as src: # opens filepath for reading
        img = src.read(masked=True) # reads image
        profile = src.profile # reads image metadata
    # transpose image shape
    img = img.transpose(1,2,0)
    return img, profile

def predict(model, modelname, image, profile, outdir):
    # predict
    predicted = model.predict(img.reshape(-1,img.shape[-1]))
    # prediction back to 2D array
    prediction = predicted.reshape(img.shape[0], img.shape[1])
    # mask nodata
    prediction = np.where(img[:,:,0] == 0, 0, prediction)
    # update metadata
#    print(profile)
    outprofile = profile.copy()
    outprofile.update(dtype='uint8',
                  count=1)
    # save
    outfile = os.path.join(outdir, modelname + '_classification_s2_2025.tif')
    with rio.open(outfile, 'w', **outprofile) as dst:
        dst.write(prediction,1)
    print('Saved predicted image to', outfile)

def pred_proba(model, modelname, image, profile, outdir, descriptions):
    # predict
    predicted = model.predict_proba(img.reshape(-1,img.shape[-1]))
    # prediction back to 2D array
    prediction = predicted.reshape(img.shape[0], img.shape[1], len(model.classes_))
    # mask nodata
    mask2d = np.any(img.mask, axis=-1)   # True where any input channel is masked
    mask2d = np.expand_dims(mask2d, axis=2)
    prediction = np.ma.where(mask2d, np.nan, prediction)
    # transpose
    prediction = prediction.transpose(2,0,1)

    if SAVE_SINGLE_BAND_PROBA: # save single bands separately
        for desc in descriptions:
            # update metadata
            outprofile = profile.copy()
            outprofile.update(dtype='float32',
                          count=1,
                          nodata=np.nan,
                          descriptions=desc)
            # choose layer from predictions
            desc_index = descriptions.index(desc)
            pred_layer = prediction[desc_index,:,:]
            # save
            outfile = os.path.join(outdir, modelname + f'_{desc}_classification_s2_2026.tif')
            with rio.open(outfile, 'w', **outprofile) as dst:
                dst.write(pred_layer,1)
            print('Saved predicted image to', outfile)

    else: # save predictions as multiband
        # update metadata
        outprofile = profile.copy()
        outprofile.update(dtype='float32',
                      count=len(model.classes_),
                      nodata=np.nan,
                      )
        # save
        outfile = os.path.join(outdir, modelname + '_classification_proba_s2_2026.tif')
        with rio.open(outfile, 'w', **outprofile) as dst:
            dst.write(prediction)
            dst.descriptions = tuple(descriptions)
        print('Saved predicted image to', outfile)
    

if __name__ == "__main__":
    img, profile = readImage(fp_img)
    model, outdir = readModel(fp_model)
    if PREDICT_PROBA:
        pred_proba(model, modelname, img, profile, outdir, descriptions)
    else:
        predict(model, modelname, img, profile, outdir)

