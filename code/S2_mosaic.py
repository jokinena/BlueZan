import argparse
import os
import glob
import numpy as np
import rasterio as rio
from rasterio.merge import merge

CLI = argparse.ArgumentParser()
CLI.add_argument('filepaths',
                 type=str,
                 nargs='+',
                 help='One or more filepaths/patterns for Sentinel-2 .SAFE folders')
CLI.add_argument('Res',
                 type=int,
                 help='Resolution of image bands to select. 10, 20 or 60')
args = CLI.parse_args()

fp_list = args.filepaths
res = args.Res

def S2_Mosaic(fp_list, res):
    if res == 10:
        bands = ['B02', 'B03', 'B04', 'B08']
    elif res == 20:
        bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    elif res == 60:
        bands = ['B01', 'B09', 'B10']
    else:
        raise ValueError("Resolution must be 10, 20, or 60")
    
    # Expand any glob patterns passed via command line
    expanded_fps = []
    for fp in fp_list:
        match = glob.glob(fp)
        if match:
            expanded_fps.extend(match)
        else:
            expanded_fps.append(fp)
            
    print(f"Processing {len(expanded_fps)} Sentinel-2 scenes for mosaic...")

    # Dictionary to store open file handles grouped by band
    band_images = {b: [] for b in bands}
    opened_srcs = []

    try:
        # Step 1: Open all files for all bands across all scenes
        for fp in expanded_fps:
            for b in bands:
                # Find the specific band file within the .SAFE structure
                search_path = os.path.join(fp, '**', 'IMG_DATA', '**', '*' + b + '_*.jp2')
                f = glob.glob(search_path, recursive=True)
                
                # Fallback for L2A structure variations (R10m, R20m, R60m folders)
                if not f:
                    search_path_res = os.path.join(fp, '**', 'IMG_DATA', '**', f'R{res}m', '*' + b + '*.jp2')
                    f = glob.glob(search_path_res, recursive=True)

                if f:
                    src = rio.open(f[0])
                    band_images[b].append(src)
                    opened_srcs.append(src)
                else:
                    print(f"Warning: Band {b} not found in {fp}")

        # Step 2: Merge each band across all scenes spatially
        merged_bands = []
        mosaic_profile = None

        for b in bands:
            if not band_images[b]:
                continue
            print(f"Merging band {b}...")
            # Merge the scene fragments for this specific band
            mosaic, output_transform = merge(band_images[b])
            merged_bands.append(mosaic)
            
            # Use the metadata from the first successfully merged band to build the profile
            if mosaic_profile is None:
                mosaic_profile = band_images[b][0].profile.copy()
                mosaic_profile.update({
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": output_transform
                })

        if not merged_bands:
            print("No valid bands were found to merge.")
            return

        # Step 3: Stack bands together along the count axis
        img_array = np.vstack(merged_bands)
        print('Mosaic Array shape:', img_array.shape)
        
        # Update profile for output
        outprof = mosaic_profile.copy()
        outprof.update(count=len(merged_bands),
                       dtype='uint32',
                       compress='DEFLATE',
                       driver='GTiff')
        
        # Define output directory and naming convention based on the first scene
        base_dir = os.path.dirname(os.path.abspath(expanded_fps[0]))
        namesplit = os.path.basename(expanded_fps[0]).split('_')
        outname = f"Mosaic_{namesplit[0]}_{namesplit[2]}_{res}m_bands.tif"
        outfile = os.path.join(base_dir, outname)
        
        # Step 4: Write out the combined mosaic GeoTIFF
        print('Writing mosaic to:', outfile)
        with rio.open(outfile, 'w', **outprof) as dst:
            dst.write(img_array)
        print('Successfully saved mosaic to', outfile)

    finally:
        # Clean up file handles safely
        for src in opened_srcs:
            src.close()
    
def main():
    S2_Mosaic(fp_list, res)

if __name__ == '__main__':
    main()
