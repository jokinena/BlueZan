# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:33:45 2025

ML prediction on underwater habitats

@author: E1008409
"""

import os 
import glob
import geopandas as gpd
import rasterio as rio
import numpy as np
import pandas as pd
from rasterio.mask import mask
from rasterio import features
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import seaborn as sns

# filepaths
fp_img = '/mnt/d/users/E1008409/MK/sansibar/planet/ChwakaBay_04-2024_psscene_analytic_8b_sr_udm2/deglint/20240430_3B_AnalyticMS_SR_8b_merge_deglint.tif'
#fp = '/mnt/d/users/E1008409/MK/sansibar/Trainings/polygons_supervised.gpkg'
fp = '/mnt/d/users/E1008409/MK/sansibar/Opetusdatat/Chwaka_annotation.gpkg'

# define output folder
outdir = os.path.join(os.path.dirname(fp), 'classification')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
# read polygons
gdf = gpd.read_file(fp, engine='pyogrio')
print(np.unique(gdf.label, return_counts=True))
# create id column
gdf['id'] = np.arange(1,len(gdf)+1,1)
# encode str labels
le = LabelEncoder()
le.fit(gdf.label)
gdf['int_class'] = le.transform(gdf.label) + 1
print(gdf.int_class)
print(np.unique(gdf.int_class))
# read image
with rio.open(fp_img) as src: # opens filepath for reading
    img = src.read() # reads image
    profile = src.profile # reads image metadata
# transpose image shape
img = img.transpose(1,2,0)
# plot image
plt.imshow(img[:,:,2:5])

#%% Simple classification
# create geometry, value pairs from polygons
geoms_values = [(geom,value) for geom, value in zip(gdf.geometry, gdf['int_class'])]
# rasterize polygons
labels = features.rasterize(geoms_values,
                       out_shape = img.shape[:2],
                       fill=0,
                       transform = profile['transform'],
                       all_touched = False,
                       default_value = 0,
                       dtype = 'uint8')
#plot labels, ie. our image annotations
plt.imshow(labels)

# exclude nodata pixels from training
train_data = img[labels > 0]
labels = labels[labels != 0]

print(train_data.shape)
print(labels.shape)

# make a train test split
X_train, X_test, y_train, y_test = train_test_split(train_data, labels,
                                                    stratify=labels,
                                                    test_size=0.3,
                                                    random_state=123)

# initialize randomForest classifier
rf = RandomForestClassifier(n_estimators=100, 
                            max_depth=None, 
                            n_jobs=5, 
                            max_features='sqrt', 
                            bootstrap=True, 
                            oob_score=True)

# fit model to training data
rf.fit(X_train, y_train)

# predict on test set 
predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)

# classification report
print(metrics.classification_report(y_test, rf.predict(X_test)))
# create confusion matrix
cm = metrics.confusion_matrix(predf['truth'], predf['predict'])

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf.classes_)
disp.plot()
plt.show()

# compute row and col sums
total = cm.sum(axis=0)
rowtotal = cm.sum(axis=1)

# create cm DataFrame
cmdf = np.vstack([cm,total])
b = np.array([[rowtotal[0]], [rowtotal[1]], [rowtotal[2]], [rowtotal.sum()]])
cmdf = np.hstack((cmdf, b))

cmdf = pd.DataFrame(cmdf, index=['Dead coral', 'Sand', 'Seagrass meadow', 'Total'],
                    columns = ['Dead coral', 'Sand', 'Seagrass meadow', 'Total'])

# print
print(pd.crosstab(predf.truth, predf.predict, margins=True))

o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
precision = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
recall = cm.diagonal() / cm.sum(axis=1) # user's accuracy

omission_errors = round((cm[1,0] + cm[2,0]) / total[0] *100, 2), round((cm[0,1] + cm[2,1]) / total[1] *100, 2), round((cm[0,2] + cm[1,2]) / total[2] *100, 2)
comission_errors = round((cm[0,1] + cm[0,2]) / rowtotal[0] *100, 2), round((cm[1,0] + cm[1,2]) / total[1] *100, 2), round((cm[2,0] + cm[2,1]) / total[2] *100, 2)


# plot 
sns.set_theme(style='white')
fig, ax = plt.subplots()
ax = sns.heatmap(cmdf, annot=True, cmap='Blues', fmt='.0f', cbar=False)
ax.xaxis.set_ticks_position('top')
ax.tick_params(axis='both', which='both', length=0)

#fig.suptitle('Bottomtype classification accuracy')
fig.suptitle('Confusion matrix')
plt.tight_layout()
#plt.savefig(os.path.join(outdir, 'rf_confusion_matrix_fin.png'), dpi=300, format='PNG')

print("Overall accuracy %.2f" % (o_accuracy))
print("Precision ", precision)
print("Recall", recall)

print('Omission errors (%)', omission_errors)
print('Comission errors (%)', comission_errors)

# dataframe with confusion matrix and the accuracies
acc_df = cmdf.copy()
acc_df.loc['Precision'] = [precision[0], precision[1], precision[2], ' ']
acc_df['Recall'] = [recall[0], recall[1], recall[2], ' ', ' ']
# save df
#acc_df.to_csv(os.path.join(outdir, 'rf_cm_acc.csv'))


#%%  Cross-validated 
# ---------------------------------------------------- # 
# Define cross-validation strategy
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
# make stratified KFolds for polygons and save them to dictionary
folds = dict()
for i, (train, test) in enumerate(skf.split(gdf.id, gdf.int_class)):
    # save train, test point_id's to dictionary
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    tr_pts = gdf['id'].iloc[train].tolist() # get point_id's by index
    te_pts = gdf['id'].iloc[test].tolist()
    folds[k] = (tr_pts, te_pts)
    print(folds.keys())
    print(folds['fold_2']) # print values from key

# plot different folds
for k in folds.keys():
    # select polygons for visualization
    tr = gdf[gdf.id.isin(folds[k][0])]
    te = gdf[gdf.id.isin(folds[k][1])]
    # create plot
    fig, ax = plt.subplots(figsize=(10,8))    
    pl1 = tr.plot(ax=ax, color='blue', label='train')
    pl2 = te.plot(ax=ax, color='orange', label='test')
    lines = [
        Line2D([0], [0], linestyle="none", marker="s", markersize=10, markerfacecolor=t.get_facecolor())
        for t in ax.collections[0:]
    ]
    labels = [t.get_label() for t in ax.collections[0:]]
    ax.legend(lines, labels)
    plt.tight_layout()
    plt.show()

# Next we select data according to each fold, train and evaluate model

# function to rasterize training data
def rasterizeData(polygons, image, image_meta):
    # create geometry, value pairs from polygons
    geoms_values = [(geom,value) for geom, value in zip(polygons.geometry, polygons['int_class'])]
    # rasterize polygons
    labels = features.rasterize(geoms_values,
                           out_shape = image.shape[:2],
                           fill=0,
                           transform = image_meta['transform'],
                           all_touched = False,
                           default_value = 0,
                           dtype = 'uint8')
    
    # exclude nodata pixels from training
    train_data = image[labels > 0]
    labels = labels[labels != 0]

    return train_data, labels

# initialize randomForest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=5 ,max_features='sqrt', bootstrap=True, oob_score=True)

result = dict({'OA': [], 'Precision': [], 'Recall': [], 'F1': []})
# CV
for k in folds:
    # select polygons according to fold split
    train = gdf[gdf.id.isin(folds[k][0])]
    test = gdf[gdf.id.isin(folds[k][1])]
    # rasterize data
    X_train, y_train = rasterizeData(train, img, profile)
    X_test, y_test = rasterizeData(test, img, profile)

    # fit training data
    rf.fit(X_train, y_train)
    # predict on test
    pred = rf.predict(X_test)
    # classification report
    print(metrics.classification_report(y_test, pred))
    # create confusion matrix
    cm = metrics.confusion_matrix(y_test, pred)   
    # compute row and col sums
    total = cm.sum(axis=0)
    rowtotal = cm.sum(axis=1)
    # compute accuracies    
    o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0)) # overall accuracy 
    precision = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
    recall = cm.diagonal() / cm.sum(axis=1) # user's accuracy
    f1_score = metrics.f1_score(y_test, pred, labels=rf.classes_, average='weighted') #f1-score
    # store metrics to dictionary
    result['OA'].append(o_accuracy)
    result['Precision'].append(precision)
    result['Recall'].append(recall)
    result['F1'].append(f1_score)
    

# get accuracy metrics
p = np.stack(result['Precision'])
r = np.stack(result['Recall'])
# sort labelnames according to sorted hab_class column
lnames = gdf[['label', 'int_class']].groupby('int_class', sort=True).max()
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
ax[1,0].set_xticklabels(lnames.label, rotation=45)
ax[1,0].set_yticks(yticks)
ax[1,0].set_yticklabels(yticks)

ax[1,1].boxplot(r)
ax[1,1].set_title('Recall')
ax[1,1].set_xticklabels(lnames.label, rotation=45)
ax[1,1].set_yticks(yticks)
ax[1,1].set_yticklabels(yticks)

plt.tight_layout()
plt.show()

#%% Predict for full image
# ---------------------------------------------------- # 
# As we now have some idea of the model performance, we can fit all the data and predict for the full image

# create geometry, value pairs from polygons
geoms_values = [(geom,value) for geom, value in zip(gdf.geometry, gdf['int_class'])]
# rasterize polygons
labels = features.rasterize(geoms_values,
                       out_shape = img.shape[:2],
                       fill=0,
                       transform = profile['transform'],
                       all_touched = False,
                       default_value = 0,
                       dtype = 'uint8')
# exclude nodata pixels from training
train_data = img[labels > 0]
labels = labels[labels != 0]

# fit data
rf.fit(train_data, labels)
# predict
predicted = rf.predict(img.reshape(-1,img.shape[-1]))
# prediction back to 2D array
prediction = predicted.reshape(img.shape[0], img.shape[1])
# mask nodata
prediction = np.where(img[:,:,0] == 0, 0, prediction)
# update metadata
print(profile)
outprofile = profile.copy()
outprofile.update(dtype='uint8',
              count=1)
# save
outfile = os.path.join(outdir, 'cv_rf_classification.tif')
with rio.open(outfile, 'w', **outprofile) as dst:
    dst.write(prediction,1)
    


























