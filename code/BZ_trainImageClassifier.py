# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 18:07:15 2025

Script to perform Machine Learning classification from annotated polygon data.

@author: E1008409
"""


import os 
import geopandas as gpd
import rasterio as rio
import numpy as np
from rasterio import features
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pickle


# filepaths
fp_img = '/mnt/d/users/e1008409/MK/sansibar/S2/Chwaka/deglint/S2_2025_chwaka_deglint.tif' # full filepath to image
fp = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/Chwaka_annotation.gpkg' # full filepath to annotation polygons

# Edit lines below to change the number of folds in cross-validation and which model to use
# set how many splits for CV
n_splits = 3
# Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=5 ,max_features='sqrt', bootstrap=True, oob_score=True)
#model = SVC(C=1, kernel='rbf', degree=3) # this can be slow! 
modelname = 'randomForest' # give name for model
labelcol = 'label' # column name which has the labels for polygon annotation


# ---------------------------------------------------- #
# No need to edit the code below

# function to rasterize training data
def createTrainData(polygons, image, image_profile):
    # create geometry, value pairs from polygons
    geoms_values = [(geom,value) for geom, value in zip(polygons.geometry, polygons['int_class'])]
    # rasterize polygons
    labels = features.rasterize(geoms_values,
                           out_shape = (image_profile['height'], image_profile['width']),
                           fill=0,
                           transform = image_profile['transform'],
                           all_touched = False,
                           default_value = 0,
                           dtype = 'uint8')
    
    # exclude nodata pixels from training
    train_data = image[labels > 0]
    labels = labels[labels != 0]

    return train_data, labels

def plotAccuracyStats(result, gdf, outdir, labelcol):
    print(result)
    # get accuracy metrics
    p = np.stack(result['Precision'])
    r = np.stack(result['Recall'])
    # sort labelnames according to sorted hab_class column
    lnames = gdf[[labelcol, 'int_class']].groupby('int_class', sort=True).max()
    yticks = [0.25,0.5,0.75,1]
    # make accuracy plot
    fig, ax = plt.subplots(2,2)
    ax[0,0].boxplot(result['OA'])
    ax[0,0].set_title('Overall accuracy')
    ax[0,0].set_xticklabels([])
    ax[0,0].set_yticks(yticks[1:])
    ax[0,0].set_yticklabels(yticks[1:])
    
    ax[0,1].boxplot(result['F1'])
    ax[0,1].set_title('F1-score')
    ax[0,1].set_xticklabels([])
    ax[0,1].set_yticks(yticks[1:])
    ax[0,1].set_yticklabels(yticks[1:])
    
    ax[1,0].boxplot(p)
    ax[1,0].set_title('Precision')
    ax[1,0].set_xticklabels(lnames[labelcol], rotation=45)
    ax[1,0].set_yticks(yticks)
    ax[1,0].set_yticklabels(yticks)
    
    ax[1,1].boxplot(r)
    ax[1,1].set_title('Recall')
    ax[1,1].set_xticklabels(lnames[labelcol], rotation=45)
    ax[1,1].set_yticks(yticks)
    ax[1,1].set_yticklabels(yticks)
    
    plt.tight_layout()
    # save plot
    plotout = os.path.join(outdir, 'CV_accuracy.png')
    plt.savefig(plotout, dpi=300)
    print('Saved plots to', plotout)
#    plt.show()

def createOutFolder(fp):
    # define output folder
    outdir = os.path.join(os.path.dirname(fp), 'classification_2025')
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    return outdir

def readPolygons(fp, labelcol):
    # read polygons
    gdf = gpd.read_file(fp, engine='pyogrio')
    # check that geometries are polygons
    assert (gdf.geom_type == 'Polygon').all(), 'Geometries are not Polygons'
    # create id column
    gdf['id'] = np.arange(1,len(gdf)+1,1)
    # encode str labels
    le = LabelEncoder()
    le.fit(gdf[labelcol])
    gdf['int_class'] = le.transform(gdf[labelcol]) + 1
    print('Unique classes and counts:', np.unique(gdf[labelcol], return_counts=True))
    return gdf

def readImage(fp_image):
    # read image
    with rio.open(fp_img) as src: # opens filepath for reading
        img = src.read() # reads image
        profile = src.profile # reads image metadata
    # transpose image shape
    img = img.transpose(1,2,0)
    return img, profile

def setCrossValidation(gdf, n_splits):
    # Define cross-validation strategy
    skf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    # make stratified KFolds for polygons and save them to dictionary
    folds = dict()
    for i, (train, test) in enumerate(skf.split(gdf.id, gdf.int_class)):
        # save train, test point_id's to dictionary
        k = 'fold_' + str(i+1)
        tr_pts = gdf['id'].iloc[train].tolist() # get point_id's by index
        te_pts = gdf['id'].iloc[test].tolist()
        folds[k] = (tr_pts, te_pts)

    return folds

# Select data according to each fold, train and evaluate model
def crossvalidateModel(model, modelname, folds, gdf, img, profile, outdir):
    result = dict({'OA': [], 'Precision': [], 'Recall': [], 'F1': []})
    # CV
    for k in folds:
        # select polygons according to fold split
        train = gdf[gdf.id.isin(folds[k][0])]
        test = gdf[gdf.id.isin(folds[k][1])]
        # rasterize data
        X_train, y_train = createTrainData(train, img, profile)
        X_test, y_test = createTrainData(test, img, profile)
    
        # fit training data
        model.fit(X_train, y_train)
        # predict on test
        pred = model.predict(X_test)
        # create confusion matrix
        cm = metrics.confusion_matrix(y_test, pred)   
        # compute row and col sums
        total = cm.sum(axis=0)
        rowtotal = cm.sum(axis=1)
        # compute accuracies    
        o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0)) # overall accuracy 
        precision = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
        recall = cm.diagonal() / cm.sum(axis=1) # user's accuracy
        f1_score = metrics.f1_score(y_test, pred, labels=model.classes_, average='weighted') #f1-score
        # store metrics to dictionary
        result['OA'].append(o_accuracy)
        result['Precision'].append(precision)
        result['Recall'].append(recall)
        result['F1'].append(f1_score)
    # save model
    model.fit(train_data, labels)
    modelout = os.path.join(outdir, modelname + '25.sav')
    pickle.dump(model, open(modelout, 'wb'))
    print('Saved model to', modelout)
    return result
    
if __name__ == "__main__":
    outdir = createOutFolder(fp)
    gdf = readPolygons(fp, labelcol)
    img, profile = readImage(fp_img)
    train_data, labels = createTrainData(gdf, img, profile)
    folds = setCrossValidation(gdf, n_splits)
    result = crossvalidateModel(model, modelname, folds, gdf, img, profile, outdir)
    plotAccuracyStats(result, gdf, outdir, labelcol)

