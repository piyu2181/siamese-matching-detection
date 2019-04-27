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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image


from utils import im_processing, tracking_utils
import lrn
import lst_tracking_config_upd_sim_stage2_global_local_interweave

class Net(nn.Module):

    def __init__(self, template_size):
        '''
        Initialization function defining the network architecture
        The initial layers are provided the same architecture as VGG-16 and will
        be initialized based on a pretrained network on ImageNet.
        '''
        super(Net, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.lrn = lrn.SpatialCrossMapLRN(1024,1024,0.5,1e-16)


        self.conv_sim = nn.Conv2d(512, 1, template_size, 1, 0)

        self.conv_sim_kernel_initialzied = False

    def forward(self, x, flag_inter_feats=False):

        x = F.max_pool2d(F.relu(self.conv1_2(F.relu(self.conv1_1(x)))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2_2(F.relu(self.conv2_1(x)))), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3_3(F.relu(self.conv3_2(F.relu(self.conv3_1(x)))))), (2, 2))
        x = F.relu(self.conv4_2(F.relu(self.conv4_1(x))))
        if flag_inter_feats: # output intermediate features, will be used for update
            y = x.clone()
        x = F.relu(self.conv4_3(x))
        x = self.lrn(x) # l2 normalize across channels

        if self.conv_sim_kernel_initialzied:
            x = self.conv_sim(x)

        if flag_inter_feats:
            return x,y

        return x


    def set_conv_sim_kernel(self, weight, bias=0):

        self.conv_sim.weight.data.copy_(weight)
        self.conv_sim.bias.data.fill_(bias)

        self.conv_sim_kernel_initialzied = True

    def reset_status(self):
        self.conv_sim_kernel_initialzied = False

    def initialize_net_from_pretrained_model(self, pretrained_model, model_name):
        '''
        Loading the weights from a pretrained VGG-16 model on ImageNet
        The pretrained model can be obtained from 
        http://isis-data.science.uva.nl/rantao/vgg16-3d698e8a.pth
        '''
        
        if model_name == 'vgg16':
            for name, params in pretrained_model.state_dict().iteritems():
                if name == 'features.0.weight':
                    self.conv1_1.weight.data.copy_(params)
                elif name == 'features.0.bias':
                    self.conv1_1.bias.data.copy_(params)
                elif name == 'features.2.weight':
                    self.conv1_2.weight.data.copy_(params)
                elif name == 'features.2.bias':
                    self.conv1_2.bias.data.copy_(params)
                elif name == 'features.5.weight':
                    self.conv2_1.weight.data.copy_(params)
                elif name == 'features.5.bias':
                    self.conv2_1.bias.data.copy_(params)
                elif name == 'features.7.weight':
                    self.conv2_2.weight.data.copy_(params)
                elif name == 'features.7.bias':
                    self.conv2_2.bias.data.copy_(params)
                elif name == 'features.10.weight':
                    self.conv3_1.weight.data.copy_(params)
                elif name == 'features.10.bias':
                    self.conv3_1.bias.data.copy_(params)
                elif name == 'features.12.weight':
                    self.conv3_2.weight.data.copy_(params)
                elif name == 'features.12.bias':
                    self.conv3_2.bias.data.copy_(params)
                elif name == 'features.14.weight':
                    self.conv3_3.weight.data.copy_(params)
                elif name == 'features.14.bias':
                    self.conv3_3.bias.data.copy_(params)
                elif name == 'features.17.weight':
                    self.conv4_1.weight.data.copy_(params)
                elif name == 'features.17.bias':
                    self.conv4_1.bias.data.copy_(params)
                elif name == 'features.19.weight':
                    self.conv4_2.weight.data.copy_(params)
                elif name == 'features.19.bias':
                    self.conv4_2.bias.data.copy_(params)
                elif name == 'features.21.weight':
                    self.conv4_3.weight.data.copy_(params)
                elif name == 'features.21.bias':
                    self.conv4_3.bias.data.copy_(params)
                else:
                    # print('skip layer %s' % name)
                    pass

        else:
            print('The net can only be initialized using vgg16!')


# This is part of whole network we want to update online.
class Net2upd(nn.Module):

    def __init__(self, kernel_size): # 'kernel_size' used to normalize sim scores
        super(Net2upd, self).__init__()

        self.conv = nn.Conv2d(512, 512, 3, 1, 1) # conv4_3
        self.lrn = lrn.SpatialCrossMapLRN(1024,1024,0.5,1e-16)

        self.normalizer_scalar = Variable(kernel_size, requires_grad=False)

    def forward(self, x1, x2):

        x1 = self.lrn(F.relu(self.conv(x1))) # weight
        x2 = self.lrn(F.relu(self.conv(x2)))

        y = F.conv2d(x2, x1)
        y = y * self.normalizer_scalar.expand_as(y)
        y = F.sigmoid(y.view(-1))
        # y = y.view(-1)

        return y


class ltsintmain:

    def __init__(self, img_query, bbox_query):
        self.use_gpu = False # use of gpu

        self.dtype = torch.FloatTensor

        self.config = lst_tracking_config_upd_sim_stage2_global_local_interweave.Config()

        # networks
        pretrained_vgg16 = models.vgg16(pretrained=False)
        pretrained_vgg16.load_state_dict(torch.load('./model_files/vgg16-3d698e8a.pth'))

        self.net_stage1 = Net(self.config.query_featmap_size_coarse)
        self.net_stage1.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')
        self.net_stage2 = Net(self.config.query_featmap_size_coarse)
        self.net_stage2.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')
        self.net_stage3 = Net(self.config.query_featmap_size_fine)
        self.net_stage3.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')


        K = torch.FloatTensor(1).fill_((float(self.config.spatial_ratio)/self.config.qimage_size_coarse)**2)

        self.net_upd = Net2upd(K)

        self.pixel_means = np.array([104.00698793, 116.66876762, 122.67891434])

        # reset net_stage2
        self.net_stage2.initialize_net_from_pretrained_model(pretrained_vgg16, 'vgg16')

        self.net_stage1.reset_status()
        self.net_stage2.reset_status()
        self.net_stage3.reset_status()

        # initialize
        self.net_upd.conv.weight.data.copy_(self.net_stage2.conv4_3.weight.data)
        self.net_upd.conv.bias.data.copy_(self.net_stage2.conv4_3.bias.data)

        print bbox_query

        self.init_box = list(bbox_query) # in x, y, width and height format

        # Get the first frame here and prepare in the format shown below
        self.init_box_w = self.init_box[2]
        self.init_box_h = self.init_box[3]
        self.init_box[2] = self.init_box[0] + self.init_box[2] - 1
        self.init_box[3] = self.init_box[1] + self.init_box[3] - 1

        self.qimg = Image.fromarray(img_query.astype('uint8'), 'RGB')


        self.qbox = self.init_box
        self.qbox[0] = self.qbox[0] - 0.5 * self.init_box_w # to include some context
        self.qbox[2] = self.qbox[2] + 0.5 * self.init_box_w
        self.qbox[1] = self.qbox[1] - 0.5 * self.init_box_h
        self.qbox[3] = self.qbox[3] + 0.5 * self.init_box_h

        # stage 1
        qimg_proc_tensor1 = im_processing.process_im_single_crop_for_network_caffe(self.qimg, self.qbox, self.config.qimage_size_coarse*2, self.config.qimage_size_coarse*2, self.pixel_means)
        qimg_proc_tensor1.unsqueeze_(0) # add one dimension to form a batch
        qimg_proc_variable1 = Variable(qimg_proc_tensor1.type(self.dtype), requires_grad=False)
        # qfeat1 = net_stage1(qimg_proc_variable1)
        qfeat1, q_inter_feats = self.net_stage1(qimg_proc_variable1,True)

        #-----#
        query_fixed_feats = q_inter_feats.data[:,:,(self.config.qimage_size_coarse//self.config.spatial_ratio)/2:(self.config.qimage_size_coarse//self.config.spatial_ratio)+(self.config.qimage_size_coarse//self.config.spatial_ratio)/2,(self.config.qimage_size_coarse//self.config.spatial_ratio)/2:(self.config.qimage_size_coarse//self.config.spatial_ratio)+(self.config.qimage_size_coarse//self.config.spatial_ratio)/2].clone()
        query_fixed_feats_var = Variable(query_fixed_feats, requires_grad=False)

        conv_sim_weight1 = qfeat1.data[:,:,(self.config.qimage_size_coarse//self.config.spatial_ratio)/2:(self.config.qimage_size_coarse//self.config.spatial_ratio)+(self.config.qimage_size_coarse//self.config.spatial_ratio)/2,(self.config.qimage_size_coarse//self.config.spatial_ratio)/2:(self.config.qimage_size_coarse//self.config.spatial_ratio)+(self.config.qimage_size_coarse//self.config.spatial_ratio)/2]
        self.net_stage1.set_conv_sim_kernel(conv_sim_weight1)

        # stage 2
        self.net_stage2.set_conv_sim_kernel(conv_sim_weight1)

        # stage 3
        qimg_proc_tensor2 = im_processing.process_im_single_crop_for_network_caffe(self.qimg, self.qbox, self.config.qimage_size_fine*2, self.config.qimage_size_fine*2, self.pixel_means)
        qimg_proc_tensor2.unsqueeze_(0) # add one dimension to form a batch
        qimg_proc_variable2 = Variable(qimg_proc_tensor2.type(self.dtype), requires_grad=False)
        qfeat2 = self.net_stage3(qimg_proc_variable2)
        conv_sim_weight2 = qfeat2.data[:,:,(self.config.qimage_size_fine//self.config.spatial_ratio)/2:(self.config.qimage_size_fine//self.config.spatial_ratio)+(self.config.qimage_size_fine//self.config.spatial_ratio)/2,(self.config.qimage_size_fine//self.config.spatial_ratio)/2:(self.config.qimage_size_fine//self.config.spatial_ratio)+(self.config.qimage_size_fine//self.config.spatial_ratio)/2]
        self.net_stage3.set_conv_sim_kernel(conv_sim_weight2)

        self.prev_box = []


    def run_ltsint(self, img_target, frame_itr):


        timg = Image.fromarray(img_target.astype('uint8'), 'RGB')
        if frame_itr % 1 == 0: #math.fmod(frame_counter+1, config.global_search_interval) == 0:
            '''
            Looks globally in the query frame using 3 levels of search
            '''
            #---------------------------STAGE 1----------------------------------------#
            timg_full_tensor = im_processing.process_frame_global_spatial_search_for_network_caffe(timg, self.init_box_w, self.init_box_h, self.config.qimage_size_coarse * self.config.reduce_factor, self.config.qimage_size_coarse*self.config.reduce_factor, self.config.spatial_ratio, self.pixel_means)
            timg_full_tensor.unsqueeze_(0)
            timg_full_var = Variable(timg_full_tensor.type(self.dtype), requires_grad=False)

            scoremap_stage1 = self.net_stage1(timg_full_var).data.cpu()

            scoremap_ = scoremap_stage1[0,0,:,:]

            overlap_factor = self.config.qimage_size_coarse / self.config.spatial_ratio / 2 - 1
            prev_score = 0.00000001
            candidates_counter = 0
            candidates_stage1 = np.zeros((self.config.num_coarse_candidates,5), dtype=np.float32)
            for ii in range(self.config.num_coarse_candidates):
                max_score, max_idx = torch.max(scoremap_.view(-1), 0)
                if candidates_counter > 0 and (max_score[0] / prev_score) < self.config.candidate_continue_threshold:
                    break

                candidates_counter = candidates_counter + 1
                prev_score = max_score[0]

                r_idx = math.ceil(float(max_idx[0]+1)/scoremap_.size(1))
                c_idx = math.fmod(max_idx[0]+1,scoremap_.size(1))
                if c_idx == 0:
                    c_idx = scoremap_.size(1)

                candidates_stage1[ii,0] = ((c_idx-1) * self.config.spatial_ratio / np.round(timg.width * self.config.qimage_size_coarse * self.config.reduce_factor / self.init_box_w) * timg.width) + 1
                candidates_stage1[ii,1] = ((r_idx-1) * self.config.spatial_ratio / np.round(timg.height * self.config.qimage_size_coarse * self.config.reduce_factor / self.init_box_h) * timg.height) + 1
                candidates_stage1[ii,2] = candidates_stage1[ii,0] + self.init_box_w - 1
                candidates_stage1[ii,3] = candidates_stage1[ii,1] + self.init_box_h - 1
                candidates_stage1[ii,4] = max_score[0]

                try:
                    scoremap_[int(np.maximum(r_idx-overlap_factor,1)-1):int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))),int(np.maximum(c_idx-overlap_factor,1)-1):int(np.minimum(c_idx+overlap_factor,scoremap_.size(1)))] = 0
                except:
                    print(int(np.maximum(r_idx-overlap_factor,1)-1), int(np.minimum(r_idx+overlap_factor,scoremap_.size(0))), int(np.maximum(c_idx-overlap_factor,1)-1), int(np.minimum(c_idx+overlap_factor,scoremap_.size(1))))


            candidates_stage1 = candidates_stage1[:candidates_counter,:]

            #---------------------------STAGE 2----------------------------------------#
            probe_regions_stage2 = tracking_utils.sample_probe_regions_multiscale_multiple_anchors(candidates_stage1[:,:4], self.config.scales_coarse, self.config.probe_factor)
            probe_regions_stage2_tensor = im_processing.process_im_multipe_crops_unordered_for_network_caffe(timg, probe_regions_stage2, self.config.qimage_size_coarse*self.config.probe_factor, self.config.qimage_size_coarse*self.config.probe_factor, self.pixel_means)
            probe_regions_stage2_var = Variable(probe_regions_stage2_tensor.type(self.dtype), requires_grad=False)

            scoremap_stage2_var, t_inter_feats_var = self.net_stage2(probe_regions_stage2_var,True)
            scoremap_stage2 = scoremap_stage2_var.data.cpu()

            #------#
            intermediate_feats = t_inter_feats_var.data.clone() #torch tensor

            max_value, s_idx, r_idx, c_idx = tracking_utils.select_max_response(scoremap_stage2)

            probe_sel = probe_regions_stage2[int(s_idx-1),:].copy()
            predicted_box_stage2 = probe_sel.copy()
            predicted_box_stage2[0] = np.maximum(probe_sel[0] + float(c_idx-1) * self.config.spatial_ratio / self.config.timage_size_coarse * (probe_sel[2]-probe_sel[0]+1), 1)
            predicted_box_stage2[1] = np.maximum(probe_sel[1] + float(r_idx-1) * self.config.spatial_ratio / self.config.timage_size_coarse * (probe_sel[3]-probe_sel[1]+1), 1)
            scale_sel = math.fmod(s_idx, self.config.scales_coarse.shape[0])
            if scale_sel == 0:
                scale_sel = self.config.scales_coarse.shape[0]
            predicted_box_stage2[2] = predicted_box_stage2[0] + float(self.init_box_w) * self.config.scales_coarse[int(scale_sel)-1] - 1
            predicted_box_stage2[3] = predicted_box_stage2[1] + float(self.init_box_h) * self.config.scales_coarse[int(scale_sel)-1] - 1


            #---------------------------STAGE 3----------------------------------------#
            probe_regions_stage3 = tracking_utils.sample_probe_regions_multiscale_single_anchor(predicted_box_stage2, self.config.scales_fine, self.config.probe_factor)
            probe_regions_stage3_tensor = im_processing.process_im_multipe_crops_ordered_for_network_caffe(timg, probe_regions_stage3, self.config.timage_size_fine, self.config.timage_size_fine, self.pixel_means)
            probe_regions_stage3_var = Variable(probe_regions_stage3_tensor.type(self.dtype), requires_grad=False)

            scoremap_stage3 = self.net_stage3(probe_regions_stage3_var).data.cpu()
            max_value, s_idx, r_idx, c_idx = tracking_utils.select_max_response(scoremap_stage3)

            probe_sel = probe_regions_stage3[int(s_idx-1),:].copy()
            predicted_box_stage3 = probe_sel.copy()
            predicted_box_stage3[0] = np.maximum(probe_sel[0] + float(c_idx-1) * self.config.spatial_ratio / self.config.timage_size_fine * (probe_sel[2]-probe_sel[0]+1), 1)
            predicted_box_stage3[1] = np.maximum(probe_sel[1] + float(r_idx-1) * self.config.spatial_ratio / self.config.timage_size_fine * (probe_sel[3]-probe_sel[1]+1), 1)
            predicted_box_stage3[2] = np.minimum(predicted_box_stage3[0] + float(predicted_box_stage2[2]-predicted_box_stage2[0]+1) * self.config.scales_fine[int(s_idx)-1] - 1, timg.width)
            predicted_box_stage3[3] = np.minimum(predicted_box_stage3[1] + float(predicted_box_stage2[3]-predicted_box_stage2[1]+1) * self.config.scales_fine[int(s_idx)-1] - 1, timg.height)

            # prev_box: for local search
            self.prev_box = predicted_box_stage3.copy()
            self.prev_box[0] = probe_sel[0] + float(c_idx-1) * self.config.spatial_ratio / self.config.timage_size_fine * (probe_sel[2]-probe_sel[0]+1)
            self.prev_box[1] = probe_sel[1] + float(r_idx-1) * self.config.spatial_ratio / self.config.timage_size_fine * (probe_sel[3]-probe_sel[1]+1)
            self.prev_box[2] = self.prev_box[0] + float(predicted_box_stage2[2]-predicted_box_stage2[0]+1) * self.config.scales_fine[int(s_idx)-1] - 1
            self.prev_box[3] = self.prev_box[1] + float(predicted_box_stage2[3]-predicted_box_stage2[1]+1) * self.config.scales_fine[int(s_idx)-1] - 1

            bbox_target = predicted_box_stage3.copy()
            return bbox_target, max_value



# if __name__ == '__main__':
#     main()
