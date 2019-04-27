'''
----------------------------------------------------------------------------------------
Author: Debjani Bhowmick, Tilburg University, 2019
----------------------------------------------------------------------------------------
Reference: This code has been obtained by modifying the code which was originally
written by Ran Tao, University of Amsterdam, and the original code is accesible at 
http://data.votchallenge.net/vot2018/trackers/LTSINT-code-2018-06-15T17_31_51.055258.zip
----------------------------------------------------------------------------------------
'''
import numpy as np

import torch

class Config():

	def __init__(self):

		# Tracker Params
		self.qimage_size_coarse = 32
		self.num_coarse_candidates =50
		self.candidate_continue_threshold = 0.5 # if the score is much smaller than previous candidate, no need to continue
		
		self.qimage_size_fine = 64
		self.probe_factor = 2
		self.timage_size_fine = self.qimage_size_fine*self.probe_factor
		self.timage_size_coarse = self.qimage_size_coarse*self.probe_factor

		self.spatial_ratio = 8
		
		self.query_featmap_size_coarse = (self.qimage_size_coarse // self.spatial_ratio)
		self.query_featmap_size_fine = (self.qimage_size_fine // self.spatial_ratio)
		self.test_featmap_size_fine = (self.timage_size_fine // self.spatial_ratio)

		#self.scales_coarse = np.array([0.2500, 0.5000, 1.0000, 1.4142, 2.8284], dtype=np.float32)
		self.scales_coarse = np.array([0.3536,0.5000,0.7071,1.0000,1.4142,2.0000,2.8284,4.0000], dtype=np.float32)
		self.scales_fine = np.array([0.7579, 0.8467, 0.9461, 1.0000, 1.1173, 1.2483], dtype=np.float32)
		

		self.scales_local_search = np.array([0.9509, 1.0000, 1.0517], dtype=np.float32)
		scale_penalty = np.full((self.scales_local_search.size, 1, 
		                              self.test_featmap_size_fine-self.query_featmap_size_fine+1, self.test_featmap_size_fine-self.query_featmap_size_fine+1), 0.96, dtype=np.float32)
		scale_penalty[2, ...] = 1.0
		self.scale_penalty = torch.from_numpy(scale_penalty)


		self.global_search_interval = 15
		self.rep_times = 10 # rep_times refer to number of times the video needs to be repeated for a long-term study

		# update
		self.niters_train = 10
		self.lr_train = 0.01 #default
		self.wd_train = 0.0005
		self.mom_train = 0.9
		self.dampening_train = 0.0
		self.interval_upd = 1
		self.PN_ratio = 0.1

		########
		self.reduce_factor = 1 # default

		#self.sa_model = './model_files/SANet_ep35_Adam-lr5e-4_ALOV250-52_seqlen10_PN0.3-0.5_pos-0.30_neg1.00-0.05_batch128.pth'
		self.seq_len = 10
		self.sa_input_dim = 81
		self.sa_thresh = 0.5

        
		# OTB100
		self.dataset = 'otb100'
		self.src_dir = ''
		self.anno_dir = ''
		self.impath_format = ''
		self.save_dir = 'webcam_temp'
		self.save_sig = 'ltsint_cam'