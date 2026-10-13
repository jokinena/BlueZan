# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 08:45:46 2025

@author: E1008409
"""

import os
import glob
import numpy as np
import geopandas as gpd
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# file
fp = '/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BZC20241107_1st_timestamp.gpkg'
fp = '/mnt/d/users/E1008409/MK/sansibar/Trainings/polygons_supervised.gpkg'
# read
gdf = gpd.read_file(fp)
# create id column
gdf['id'] = np.arange(1,len(gdf)+1,1)

# print unique values
print(np.unique(gdf.habitat, return_counts=True))

# make KFold split
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

def my_plot(train_points, test_points):
    
    ax1 = train_points.plot(ax=ax, color='blue')
    ax1 = test_points.plot(ax=ax, color='orange')
  
    return [ax1]

# plot different folds
fig, ax = plt.subplots(figsize=(10,8))
handles, labels = ax.get_legend_handles_labels()
artists = []
for i, (train, test) in enumerate(kfold.split(gdf.habitat, gdf.hab_class)):
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    print(k)
    tr_pts = gdf[gdf.index.isin(train)] # get point_id's by index
    te_pts = gdf.iloc[test]
    print(np.unique(tr_pts.habitat, return_counts=True))
    uniq = np.unique(tr_pts.habitat, return_counts=True)
    print(round(uniq[1][0]/len(tr_pts)*100,1 ), round(uniq[1][1]/len(tr_pts)*100, 1), round(uniq[1][2]/len(tr_pts)*100,1), round(uniq[1][3]/len(tr_pts)*100,1))
    
    pl1 = my_plot(tr_pts, te_pts)
    plt.tight_layout()
    artists.append(pl1)

patch1 = mpatches.Patch(color='blue', label='Train')
patch2 = mpatches.Patch(color='orange', label='Test')
handles.extend([patch1, patch2])
fig.legend(handles=handles)
ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500)    
plt.show()

# ----------------------------------------------------------- #
outdir = os.path.join(os.path.dirname(fp), 'plots')
if not os.path.isdir(outdir):
    os.mkdir(outdir)
skf = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
# make stratified KFolds for polygons and save them to dictionary
folds = dict()
for i, (train, test) in enumerate(skf.split(gdf.id, gdf.hab_class)):
    # save train, test point_id's to dictionary
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    tr_pts = gdf[gdf.index.isin(train)] # get point_id's by index
    te_pts = gdf['id'].iloc[test].tolist()
    print(np.unique(tr_pts.habitat, return_counts=True))
    uniq = np.unique(tr_pts.habitat, return_counts=True)
    print(round(uniq[1][0]/len(tr_pts)*100,1 ), round(uniq[1][1]/len(tr_pts)*100, 1), round(uniq[1][2]/len(tr_pts)*100,1), round(uniq[1][3]/len(tr_pts)*100,1))

    folds[k] = (tr_pts, te_pts)
    print(folds.keys())
    print(folds['fold_1']) # print values from key

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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(lines, labels)
    plt.suptitle(k)
    plt.tight_layout()
    fileout = os.path.join(outdir, 'plot_' + k + '.png')
    plt.savefig(fileout, dpi=300, format='PNG')
    plt.show()

import imageio
filenames = [f for f in glob.glob(os.path.join(outdir, '*.png'))]
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(os.path.join(outdir, 'stratCV_plot.gif'), images, loop=0, duration=1000)    
    
# ----------------------------------------------------------- #
# find randomly sampled pixel locations
my_idcs = np.where(labels.reshape(-1))[0]
my_idx = np.random.choice(my_idcs, int(len(my_idcs)*0.3))
temp = np.where(labels != 0, 1, 0)
temp = temp.reshape(-1)
temp[my_idx] = 2
temp = temp.reshape(labels.shape)
plt.imshow(temp)
outprofile = profile.copy()
outprofile.update(dtype='uint8',
              count=1)
# save
outfile = os.path.join(outdir, 'random_choice.tif')
with rio.open(outfile, 'w', **outprofile) as dst:
    dst.write(temp,1)
        