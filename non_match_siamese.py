#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 07:16:53 2019
Script: This is for applying seimese matching filter 
Output: Output is a .dat file having informations about the frame with its number,
        matched bounding box co-ordinates and simillarity score for each farme
@author: Debjani Bhowmick
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


#-------------------------------------------------------------------        
def show_frames(mirror=False):
    # Path for the collection of frames to match and outputs 
    label_dir  = "../../data/labelled_data/final_label/"
    file = input("give the file name ")
    ref_file = input("give the ref file name ")
    file_path = os.path.join(label_dir, file)
    ref_file_path = os.path.join(label_dir, ref_file)
    col_names = ["frame_id", "x_min", "x_max", "y_min", 'y_max']
    df = pd.read_csv(file_path, sep="\t", names= col_names, index_col= None)
    df = df.sort_values(by = ["frame_id"])
    df_ref = pd.read_csv(ref_file_path, sep="\t", names= col_names, index_col= None)
    num_rows = (df.shape)[0]
    
    # copying files to correspondig folder
    image_dataset_dir = "../../data/images/resized_images/"
    matching_path = "../../data/refrigarator/"
    #matching_path = "../../data/magazine/"
    path = "../../data/"
    output_dir = "../../data/siamese_output/"
    data_dir = os.path.join(path,file[:-4])
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        for frame in df[0:num_rows - 1]["frame_id"]:
            fname = str(int(frame)).zfill(5) + ".jpg"
            src = os.path.join(image_dataset_dir, fname)
            dst = os.path.join(data_dir, fname)
            shutil.copyfile(src, dst)
    # making a list containing all reference ground truths
    count = 0
    frame_list = []
    while count < 1:
        frame_input = input("Give choosen frames")
        frame_list.append(int(frame_input))
        count += 1
    
    # preparing file where to store the output.------------------------------
    out1 = input("give the file name where you want to save the output ")
    out_dir = os.path.join(output_dir,out1)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for frame in frame_list:
        temp = "match_with_" + str(frame)
        out_dir1 = os.path.join(out_dir,temp)
        if not os.path.exists(out_dir1):
            os.mkdir(out_dir1)
        fname = file[:-4] + str(frame)
        fname = os.path.join(out_dir1,fname)
        print("matching score will be stored in ", fname + ".dat")
        fname = fname + ".dat"
        fid = open(fname, "w")
        
        # Getting names of all files in the folder
        get_files_path = data_dir + "/*.*"
        fpath_list = sorted(glob.glob(get_files_path))
        
        # Getting names of all files in the folder
        get_files_path = data_dir + "/*.*"
        fpath_list = sorted(glob.glob(get_files_path))
        frame_itr = 0 # give frame no
        no_frames = len(fpath_list)
        
        img = matching_path +  str(frame).zfill(5) + ".jpg"
        print(img)
        first_img = cv2.imread(img)
        bbox = []
        bbox_temp = df_ref[df_ref["frame_id"] == frame].iloc[:,1:].values
        for item in bbox_temp:
            for i in item:
            #writting the informations in reference file
                bbox.append(i)
            print(bbox)
        obj_ltsint = ltsint.ltsintmain(first_img, bbox)
        
        while frame_itr < (num_rows -1) :
            frame_id =df["frame_id"][frame_itr]
            frid_path = fpath_list[0][:-9]
            print(frid_path)
            frid_path = frid_path + str(int(frame_id)).zfill(5) + ".jpg"
            
            img = cv2.imread(frid_path)
            #img = img[0::2, 0::2, :]
            ## Tracking component to be added here
            bbox_ltsint, sim_score = obj_ltsint.run_ltsint(img, frame_itr)
            print(frame_id, sim_score)
            print bbox_ltsint
            bbox[0] = bbox_ltsint[0]
            bbox[1] = bbox_ltsint[1]
            bbox[2] = bbox_ltsint[2] - bbox[0] + 1
            bbox[3] = bbox_ltsint[3] - bbox[1] + 1
            frame_itr += 1
            fid.write(str(frame_id) + "\t" + str(bbox[0]) + "\t" + str(bbox[1]) +  "\t" + str(bbox[2]) + "\t" + str(bbox[3]) + "\t"+ str(int(sim_score)) + "\n" )
        
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            #print p1, p2
            #cv2.rectangle(img, p1, p2, (255, 0, 0), 4, 1)
            #cv2.putText(img, 'Frame: ' + str(frame_itr+1) + 'Sim: ' + str(int(sim_score)), (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
            #cv2.imshow('tracking', img)
            #wait = input('Press something to continue')
            #print('Continued')
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        
        print('Exiting')
    #sys.exit(0)


def main():
    show_frames(mirror=True)


if __name__ == '__main__':
    main()                
                