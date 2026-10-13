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


# filepaths
fp_img = '/mnt/d/users/e1008409/MK/sansibar/planet/ChwakaBay_04-2024_psscene_analytic_8b_sr_udm2/deglint/20240430_3B_AnalyticMS_SR_8b_merge_deglint.tif' # full filepath to image to be classified
fp_img = '/mnt/d/users/e1008409/MK/sansibar/S2/Chwaka/deglint/S2_2025_chwaka_deglint.tif' # full filepath to image to be classified
fp_model = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/classification_2025/randomForest25.sav' # full filepath to model

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
        img = src.read() # reads image
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
 
if __name__ == "__main__":
    img, profile = readImage(fp_img)
    model, outdir = readModel(fp_model)
    predict(model, modelname, img, profile, outdir)

