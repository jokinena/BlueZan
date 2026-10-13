# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:18:18 2025



@author: E1008409
"""

import argparse
import geopandas as gpd

 
CLI = argparse.ArgumentParser()
CLI.add_argument('filepath',
                 type=str,
                 help='Filepath for field data points')
args = CLI.parse_args()
 

# filepath
#fp = '/mnt/d/users/e1008409/MK/sansibar/Mission_10_2024/Field_work/BlueZan_habitat_data_11_2024.gpkg'

def findMaxColumn(fp):
    # read
    gdf = gpd.read_file(fp)
    # bottom type columns 
    subset_cols = ['Mixed', 'Coral', 'Dead coral', 'Seagrass meadow', 'Coral rag', 'Macroalgae', 'Sand']
    # columns to float
    gdf[subset_cols] = gdf[subset_cols].astype(float)
    # replace NaN with 0
    gdf = gdf.fillna(0)
    # find which bottom type column has the highest value
    gdf['label'] = gdf[gdf.columns[21:28]].idxmax(axis=1, skipna=True)
    # save geodataframe
    print(gdf[subset_cols+['label']]) # comment out this row if you don't want to print the overview of the result
    outname = fp.split('.')[0] + '_labels.gpkg'
    gdf.to_file(outname)
    print('Saved file to:', outname)
    
def main():
    findMaxColumn(args.filepath)

if __name__=="__main__":
    main()


