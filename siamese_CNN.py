#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:04:15 2019
Script: This is for applying seimese matching filter 
Output: Output is a .dat file having informations about the frame with its number,
        matched bounding box co-ordinates and simillarity score for each farme
@author: Debjani Bhowumick
"""
# ----------------Importing required packages---------------------------
import numpy as np
import pandas as pd
import ltsint
import shutil
import glob
import sys
import cv2
import os




def show_frames(mirror=False):
    #-------------------------------------------------------------------  
    # Path for the collection of frames to match and outputs     
    original_dataset_dir = "../../data/images/resized_images/"
    label_dir  = "../../data/labelled_data/final_label/"
    output_dir = "../../data/siamese_output/"
    file = input("give the file name")
    file_path = os.path.join(label_dir, file)
    col_names = ["frame_id", "x_min", "x_max", "y_min", 'y_max']
    df = pd.read_csv(file_path, sep="\t", names= col_names, index_col= None )
    df = df.sort_values(by = ["frame_id"])
    frame_list = []
    num_rows = (df.shape)[0]
    
    # To save reference bounding box--------------------------------------
    ref_file = label_dir + file[:-4] + "_ref" + ".dat"
    ref_fid = open(ref_file, "w")
    
    # copying files to their correspondig folder--------------------------------
    path = "../../data/"
    data_dir = os.path.join(path,file[:-4])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    for frame in df[0:num_rows - 1]["frame_id"]:
        fname = str(int(frame)).zfill(5) + ".jpg"
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(data_dir, fname)
        shutil.copyfile(src, dst)
        
    # Making a list containing all reference ground truths------------------
    count = 0
    while count < 1:
        frame_input = input("Give choosen frames")
        frame_list.append(int(frame_input))
        count += 1
        
    # Preparing file where to store the output.------------------------------
    for frame in frame_list:
        out_dir = os.path.join(output_dir, file[:-4])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        temp = "match_with_" + str(frame)
        out_dir1 = os.path.join(out_dir,temp)
        if not os.path.exists(out_dir1):
            os.mkdir(out_dir1)
        fname = file[:-4] + str(frame)
        fname = os.path.join(out_dir1,fname)
        print("matching score will be stored in ", fname + ".dat")
        fname = fname + ".dat"
        fid = open(fname, "w")
        
        
        # Getting names of all files in the folder---------------------------
        get_files_path = data_dir + "/*.*"
        fpath_list = sorted(glob.glob(get_files_path))

        # give frame no from where you want to matching
        frame_itr = 450
        no_frames = len(fpath_list)
        
        ref_img = data_dir + "/" +  str(frame).zfill(5) + ".jpg"
        first_img = cv2.imread(ref_img)
        bbox = []
        #collecting bounding box from the reference ground truth
        bbox_temp = df[df["frame_id"] == frame].iloc[:,1:].values
        ref_fid.write(str(frame))
        for item in bbox_temp:
            for i in item:
            #writting the informations in reference file
                ref_fid.write("\t" + str(i))
                bbox.append(i)
        print(bbox)
        ref_fid.write("\n")
        obj_ltsint = ltsint.ltsintmain(first_img, bbox)
        while frame_itr < (num_rows-1) :
            frame_id =df["frame_id"][frame_itr]
            frid_path = fpath_list[0][:-9]
            print(frid_path)
            frid_path = frid_path + str(int(frame_id)).zfill(5) + ".jpg"
            print(frame_id)
            img = cv2.imread(frid_path)
            ## Tracking component to be added here
            bbox_ltsint, sim_score = obj_ltsint.run_ltsint(img, frame_itr)
            print bbox_ltsint
            bbox[0] = bbox_ltsint[0]
            bbox[1] = bbox_ltsint[1]
            bbox[2] = bbox_ltsint[2] - bbox[0] + 1
            bbox[3] = bbox_ltsint[3]  - bbox[1] + 1
            frame_itr += 1
            fid.write(str(frame_id) + "\t" + str(bbox[0]) + "\t" + str(bbox[1]) +  "\t" + str(bbox[2]) + "\t" + str(bbox[3]) + "\t"+ str(int(sim_score)) + "\n" )
             
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            print frame_itr, sim_score
            print p1, p2
            
            '''plotting and monitoring the matching '''
            
            #cv2.rectangle(img, p1, p2, (255, 0, 0), 4, 1)
            #cv2.putText(img, 'Frame: ' + str(frame_itr+1) + 'Sim: ' + str(int(sim_score)), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            #cv2.imshow('tracking', img)
            #wait = input('Press something to continue')
            #print('Continued')
            # On pressing escape
            if cv2.waitKey(1) == 27:
                break  # esc to quit
            
        print('Exiting')
    #sys.exit(0)


def main():
    show_frames(mirror=True)


if __name__ == '__main__':
    main()
    
    