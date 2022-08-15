# import the data organization file and work through each file in the pattern and shove each frame into the right directory


#read the data org and get unique colors

# create the directories for the colours

# work down the list of things that have a bitname 

#Each one will have an imaging round and frame number to extract as well as fiducials

import pandas as pd
from pathlib import Path

import os 
import numpy as np
import skimage.io as skio
from wand.image import Image



def split(fov, original_img_dir, dir_488, dir_650, dir_568, clahe_adj=False):
    
    # There are 9 imaging rounds in total, but we only use the 0-7, ignoring index 8 and 9.
    ir = np.array([0,1,2,3,4,5,6,7])
    
    ir_str = [str(i).zfill(2) for i in ir]

    fov_str = str(fov).zfill(3)
    
    # Slice number
    even_num = np.array([0,2,4,6,8,10,12,14,16,18])
    
    for ir in ir_str:
        
        img_path = os.path.join(original_img_dir, 'merFISH_merged_' + ir + '_' + fov_str + '.tif')
        
        img = np.array(skio.imread(img_path), dtype=np.uint16)
        
        for j, e in enumerate(even_num):
            
            
            if clahe_adj == False:

                skio.imsave(os.path.join(dir_650, f'merFISH_{ir}_{fov_str}_{str(j+1).zfill(2)}.tiff'), img[e])
                
                skio.imsave(os.path.join(dir_568, f'merFISH_{ir}_{fov_str}_{str(j+1).zfill(2)}.tiff'), img[e+1])
                
            else:
            
                # Converts from numpy array to Wand Image
                wi_ch650 = Image.from_array(img[e])
                
                wi_ch568 = Image.from_array(img[e+1])
                        
                # Invoke clahe function
                wi_ch650.clahe(40, 40, 256, 3)
                
                wi_ch568.clahe(40, 40, 256, 3)

                # Save the image
                wi_ch650.save(filename = os.path.join(dir_650, f'merFISH_{ir}_{fov_str}_{str(j+1).zfill(2)}.tiff'))
                        
                wi_ch568.save(filename = os.path.join(dir_568, f'merFISH_{ir}_{fov_str}_{str(j+1).zfill(2)}.tiff'))
            
        
            #Storing the beads
            skio.imsave(os.path.join(dir_488, f'merFISH_{ir}_{fov_str}_{str(j+1).zfill(2)}.tiff'), img[20].astype(np.uint16))




img_dir = os.path.join(os.getcwd(), 'merfish_cambridge_reorganized_and_clahe_2')

original_img_dir = os.path.join(os.getcwd(), 'merfish_cambridge_all_fovs_0')

dir_488 = os.path.join(img_dir, '488nm, Raw')

dir_650 = os.path.join(img_dir, '650nm, Raw')

dir_568 = os.path.join(img_dir, '568nm, Raw')

Path(dir_488).mkdir(parents=True, exist_ok=True)

Path(dir_650).mkdir(parents=True, exist_ok=True)

Path(dir_568).mkdir(parents=True, exist_ok=True)

def splitter(num_fov):
    
    for fov in range(num_fov):
    
        split(fov, original_img_dir, dir_488, dir_650, dir_568, clahe_adj=True)
        
splitter(69)
