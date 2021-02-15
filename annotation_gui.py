#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:56:25 2021

@author: jimmytabet
"""

############################## SET UP ANNOTATOR ###############################

# imports
import sys
import os
import glob
import numpy as np
from scipy import ndimage
import cv2

#----------------------------------USER INPUT---------------------------------#

# if working in terminal, run:     'python /path/to/annotation_gui.py /path/to/data'
#   or cd to data folder and run: 'python /path/to/annotation_gui.py $(pwd)'
#   or edit path_to_data and run: 'python /path/to/annotation_gui.py'

# if working in Jupyter Notebook, run '%load annotation_gui.py' in new .ipynb file and change JN = True
JN = False

# path to data (optionally passed in terminal - use '$(pwd)' to pass pwd)
path_to_data = '/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/ANNOTATOR TEST/Cellpose_tiles'

# labels dictionary
#!!!!!!!!!!!! WARNING, KEYS MUST BE UNIQUE AND 0-9 (EXCEPT 'EDGE') !!!!!!!!!!!#
labels_dict = {-1: 'edge', # automatically detected, key can be any (unique) int
                0: 'unknown',
                1: 'blurry',
                2: 'interphase',
                3: 'prophase',
                4: 'metaphase',
                5: 'anaphase',
                6: 'telophase'}

# show label dictionary key(s)
show_label = ['l','d']

# manually assign edge
man_edge = ['-']

# back key(s)
back_key = ['b']

# exit key(s)
exit_key = ['q']

# threshold percentage of area needed to see edge cell/automatically assign edge cell
edge_area_thresh = 0.7

# half of size to show cell
cell_half_size = 100

# half of size to show tile (must be < 800/2=400)
# set <= 0 to display whole tile
show_half_size = 0 # < 400

#-----------------------------------------------------------------------------#

# get edge key for automatic edge assignment
edge_key = [k for k,v in labels_dict.items() if v == 'edge'][0]

# set path_to_data if run in terminal
if len(sys.argv) > 1 and not JN:
    path_to_data = sys.argv[1]
else:
    path_to_data = path_to_data
    
# check that given path is actually a folder
while not os.path.isdir(path_to_data):
    path_to_data = input('ERROR: '+path_to_data+' not found, try again\npath to data: ')
    path_to_data = os.path.abspath(path_to_data)

# change directory to data
os.chdir(path_to_data)
# create list of tiles to annotate (exclude 'finished' tiles)
tiles = sorted(glob.glob(os.path.join(path_to_data,'**/*.npy'), recursive=True))
tiles = [file for file in tiles if not 'finished' in file]

# folder for annotation results
results_folder = os.path.join(path_to_data, 'annotation_results')

# show annotator set up if there are files to annotate
if tiles:

    # raise error if key conflict/duplicate keys found
    all_keys = [str(i) for i in [labels_dict.keys(), show_label, man_edge, back_key, exit_key] for i in i]
    key_names = ['labels_dict', 'show_label', 'man_edge', 'back_key', 'exit_key']
    if len(all_keys) != len(set(all_keys)):
        raise ValueError('Key conflict/duplicate keys found, check '+'/'.join(key_names)+' variables!')
    
    # make sure show_half_size behaves
    while show_half_size >= 400:
        print('WARNING: show_half_size (half of tile to show) must be less than 800/2 = 400', end='')
        show_half_size = input('enter new value for show_half_size (or 0 to show entire tile): ')
        isint = False
        while not isint:
            try:
                show_half_size = int(show_half_size)
                isint = True
            except ValueError:
                show_half_size = input('please enter integer: ')
    
    # if show_half_size is smaller than cell_half_size, revert to showing entire tile
    if 0 < show_half_size < cell_half_size:
        show_half_size = 0
        print('show_half_size too small, reverting to showing entire tile')
    
    # create results folder if does not exsist
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
        print('created annotation results folder: ', results_folder)
    
    # print annotator set up
    print()
    print('ANNOTATOR SET UP:')
    print()
    
    # data and results folders
    print('data folder (pwd):\n\t', os.getcwd())
    print()
    print('annotation results folder:\n\t', os.path.join(os.getcwd(), results_folder))
    print()
    
    # files to annotate
    print('files ('+str(len(tiles))+' total):')
    short_name = [os.path.relpath(tile) for tile in tiles]
    for i in short_name[:5]:
        print('\t',i)
    if len(tiles) > 5:
        print('\t ...')
    print()

    # exit key(s)
    print('exit key(s):\n\t', exit_key)
    print()
    
    # back key(s)
    print('back key(s):\n\t', back_key)
    print()

    # manually assign edge
    print('manually assign edge key(s):\n\t', man_edge)
    print()

    # show label key(s)
    print('show label dictionary key(s):\n\t', show_label)
    print()

    # labels
    print('labels:')
    for k,v in labels_dict.items():
        if v=='edge':
            print('\t%2d: %s (automatically assigned)' %(k,v))
        else:
            print('\t%2d: %s' %(k,v))
    print()
    
############################### RUN ANNOTATOR #################################

# close all windows
cv2.destroyAllWindows()

################## LOOP THROUGH EVERY 800X800 TILE IN FOLDER ##################

# init exited to see if user exited annotator
exited = False

for tile in tiles:
    
    # print current tile
    print('-'*53)
    print('\''+os.path.basename(tile).upper()+'\'')

    # load Cellpose data (raw image, masks, and outlines)
    data = np.load(tile, allow_pickle=True).item()
    raw = data['img']
    masks = data['masks']
    outlines = data['outlines']
    
    # find number of identified cells
    num_masks = np.max(masks)

    # init repeat and correct (to catch if number of labels does not equal number of cells)
    repeat = False
    correct = False
    
############# LOOP UNTIL NUMBER OF LABELS MATCHES NUMBER OF CELLS #############

    while not correct:
        
        # init cell label and list for cell labels in tile
        label = None
        labels=[]
        
        # init mask_id/cell number
        mask_id = 1
        
####################### LOOP THROUGH EVERY CELL IN TILE #######################

        while mask_id <=num_masks:
            
            # print cell ID
            print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')

            # find center of mass (as integer for indexing)
            center = ndimage.center_of_mass(masks==mask_id)
            center = np.array(center).astype(int)
            
            # find bounding box indices for showing isolated cell
            r1 = max(0, center[0]-cell_half_size)
            r2 = min(raw.shape[0], center[0]+cell_half_size)
            c1 = max(0, center[1]-cell_half_size)
            c2 = min(raw.shape[1], center[1]+cell_half_size)
        
            # check bounding box to see if cell on edge            
            area_ratio = (r2-r1)*(c2-c1)/(cell_half_size*2)**2
    
#-------------automatically classify as edge if too close to edge-------------#

            if area_ratio < edge_area_thresh:
                label = int(edge_key)
                # add 'auto' designation for when later trying to go back
                labels.append(str(label) + 'auto')
                print(labels_dict[label]+' (automatically assigned)')
                mask_id += 1
                continue
            
            # fix edge case if most of area is in frame
            else:
                rfix = cell_half_size*2 - (r2-r1)
                cfix = cell_half_size*2 - (c2-c1)
                if r1 == 0: r2 += rfix
                if r2 == raw.shape[0]: r1 -= rfix
                if c1 == 0: c2 += cfix
                if c2 == raw.shape[1]: c1 -= cfix
    
            # find bounding box indices for showing tile if given size
            if show_half_size:                
                show_r1 = max(0, center[0]-show_half_size)
                show_r2 = min(raw.shape[0], center[0]+show_half_size)
                show_c1 = max(0, center[1]-show_half_size)
                show_c2 = min(raw.shape[1], center[1]+show_half_size)
                show_rfix = show_half_size*2 - (show_r2-show_r1)
                show_cfix = show_half_size*2 - (show_c2-show_c1)
                if show_r1 == 0: show_r2 += show_rfix
                if show_r2 == raw.shape[0]: show_r1 -= show_rfix
                if show_c1 == 0: show_c2 += show_cfix
                if show_c2 == raw.shape[1]: show_c1 -= show_cfix
            else:
                show_r1 = 0
                show_r2 = raw.shape[0]
                show_c1 = 0
                show_c2 = raw.shape[1]
            
#--------------------------construct images to show---------------------------#

            # copy raw to show outline
            raw_outline = raw.copy()
            raw_outline[outlines==mask_id] = raw.max()
            raw_outline = raw_outline[show_r1:show_r2,show_c1:show_c2]
            # copy raw to show isolated cell
            raw_isolate = raw.copy()
            raw_isolate[masks!=mask_id] = 0
            # resize isolated cell to show with tile
            raw_isolate = raw_isolate[r1:r2,c1:c2]
            raw_isolate = cv2.resize(raw_isolate, raw_outline.shape)
            
            # shorthand for arranging images
            left_img = raw_outline
            right_img = raw_isolate
            
            # add white border between panels
            col = 5
            border = raw.max()*np.ones([left_img.shape[0], col])
            left_img = np.concatenate([left_img, border], axis=1)
            
            # stitch left and right windows together
            together = np.concatenate([left_img/left_img.max(), right_img/right_img.max()], axis=1)

#---------------------------set up annotator window---------------------------#

            # show cell number in window with extra info as necessary
            window = os.path.basename(tile).upper()+': CELL '+str(mask_id)+' OF '+str(num_masks)
            if repeat:
                window = 'ERROR: NUMBER OF LABELS DID NOT MATCH NUMBER OF CELLS, REPEATING '+os.path.basename(tile).upper()
                repeat = False
            elif label == int(edge_key):
                window += ' (previous cell(s) on edge)' 
                
            # show annotator window and image
            cv2.namedWindow('Cell Annotator', cv2.WINDOW_AUTOSIZE)
            cv2.setWindowTitle('Cell Annotator', window)
            cv2.imshow('Cell Annotator', together)
            
            # cv2.resizeWindow('Cell Annotator', together.shape[1], together.shape[0]) 
    
#----------------------------------annotate-----------------------------------#
            
            # init back (to allow for returning to previous cell)
            back = False
            
            # init valid to check if response is valid
            valid = False
            
################# LOOP UNTIL VALID LABEL RESPONSE IS DETECTED #################

            while not valid:
        
                # wait for user entry and store
                res = cv2.waitKey(0)
                label = chr(res)

#----------------------------------exit key-----------------------------------#

                # if exit key is pressed, activate exited condition and break out of valid loop
                if label in exit_key:
                    exited = True
                    print('\n')
                    print('-'*53)
                    print('EXITING ANNOTATOR...')
                    break
                
#---------------------------show labels dictionary----------------------------#

                elif label in show_label:
                    print('\n')
                    # print(labels_dict)
                    print('labels:')
                    for k,v in labels_dict.items():
                        if v=='edge':
                            print('\t%2d: %s (automatically assigned)' %(k,v))
                        else:
                            print('\t%2d: %s' %(k,v))
                    print()
                    print('manually assign edge key(s): ', man_edge)
                    print('back key(s): ', back_key)
                    print('exit key(s): ', exit_key)
                    print()
                    print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                
#----------------------------------back key-----------------------------------#                
                
                # if back key is pressed, attempt to go back one cell
                elif label in back_key:
                    # reverse through labels to check if possible
                    for index, lab in enumerate(labels[::-1]):    
                        # if possible, reset to that cell
                        if lab != str(int(edge_key)) + 'auto':
                            # activate back condition and valid input
                            back = True
                            valid = True
                            # reset cell number
                            mask_id = len(labels) - index
                            # reset labels
                            labels = labels[:mask_id-1]
                            print('BACK TO CELL', mask_id)
                            break

                    # if not possible, raise error
                    if not back:
                        print('UNABLE TO RETURN')
                        print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                        cv2.setWindowTitle('Cell Annotator', 'UNABLE TO RETURN') 

#------------------------key not in labels dictionary-------------------------#

                # if label is not in labels dictionary, raise error
                elif label not in [str(i) for i in labels_dict.keys()] + man_edge:
                    print('unrecognized label')
                    print('cell: %2d of %2d --> class: ' % (mask_id, num_masks), end='')
                    cv2.setWindowTitle('Cell Annotator', 'ERROR: UNRECOGNIZED LABEL')    
                
#--------------------------------valid label----------------------------------#

                # add label if valid
                else:
                    # activate valid input
                    valid = True
                    if label in man_edge:
                        label = int(edge_key)
                        print(labels_dict[label]+' (manually assigned)')
                    else:
                        label = int(label)
                        print(labels_dict[label])
                    labels.append(label)

#--------------------------response to valid label----------------------------#

            # if exit key is pressed in valid loop, break out of cell loop
            if label in exit_key:
                break
            # if back is triggered, reset to false and revert to previous cell
            elif back:
                back = False
            # continue to next cell if everything is fine
            else:
                mask_id += 1
        
#--------------check if number of labels match number of cells----------------#
        
        # if exit key is pressed in cell loop, break out of correct loop   
        if label in exit_key:
            break
    
        # raise error if there is not a label for every cell in tile and repeat tile annotation
        if len(labels) != num_masks:
            # activate repeat condition
            repeat = True
            print()
            print('ERROR: number of labels does not match number of cells, repeating \''+os.path.basename(tile)+'\' ...')
            print()
        
        # fix and save labels and move files if all is good
        else:            
            # activate correct condition
            correct = True
            
            # fix automatic edge label
            for i,lab in enumerate(labels):
                if lab == str(int(edge_key)) + 'auto':
                    labels[i] = int(edge_key)
                else:
                    pass
            
            # save and move files
            # get file name, position/data paths, and annotated data path
            file_name = os.path.basename(tile)
            position_path = os.path.dirname(tile)
            position_folder = os.path.basename(position_path)            
            data_folder = os.path.basename(os.path.dirname(position_path))
            annotated_data_path = os.path.join(results_folder,data_folder)
            
            # get finished and results paths
            results_path = os.path.join(annotated_data_path, position_folder, file_name[:-7]+'annotated.npz')
            finished_path = os.path.join(position_path+'_finished', os.path.basename(tile))
           
            # create results and finished folders if not already created
            # new annotated data folder
            if not os.path.isdir(annotated_data_path):
                os.makedirs(annotated_data_path)
            # new annotated position folder
            if not os.path.isdir(os.path.dirname(results_path)):
                os.makedirs(os.path.dirname(results_path))
            # finished position folder
            if not os.path.isdir(os.path.dirname(finished_path)):
                os.makedirs(os.path.dirname(finished_path))

            # save annotated info for tile once all cells have been labeled
            np.savez(results_path, raw=raw, masks=masks, labels=labels, labels_dict = labels_dict)
            # move file to completed folder
            os.replace(tile, finished_path)

            # print paths
            short_path = os.path.relpath(tile)
            short_results = os.path.relpath(results_path)     
            short_finished = os.path.relpath(finished_path)            
            print()
            print('FINISHED --> \''+short_path+'\'')
            print(' RESULTS --> \''+short_results+'\'')
            print('    TILE --> \''+short_finished+'\'')
            print()

#---------------------------------(exit key)----------------------------------#

    # if exit key is pressed in correct loop, break out of tile loop   
    if label in exit_key:
        break
    
############################### CLOSE ANNOTATOR ###############################
    
# close all windows
cv2.destroyAllWindows()

# print exit message
if not exited:
    print('-'*53)
    print('NO FILES FOUND/ALL FILES FINISHED IN:\n'+path_to_data)
else:
    print('ANNOTATOR EXITED')