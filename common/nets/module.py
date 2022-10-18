import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.cfg_utils import (getBooleanFromCfg, getFloatFromCfg, getIntFromCfg,
                             getStringFromCfg)
from utils.human_models import flame, mano, smpl
from utils.transforms import sample_joint_features, soft_argmax_3d

from nets.layer import (ConvBNReLU, DWConvLayer, make_conv_layers,
                        make_dwconv_layers, make_linear_layers)


class PositionNet(nn.Module):
    def __init__(self, cfg, in_channel):
        super(PositionNet, self).__init__()
        self.cfg = cfg
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        if getBooleanFromCfg(cfg, 'model_position_use_dw'):
            self.conv = make_dwconv_layers([in_channel,self.joint_num*self.cfg.output_hm_shape[0]], kernel=3, stride=1, padding=1, bnrelu_final=False)
        else:
            self.conv = make_conv_layers([in_channel,self.joint_num*self.cfg.output_hm_shape[0]], kernel=1, stride=1, padding=0, bnrelu_final=False)

    def forward(self, img_feat):
        joint_hm = self.conv(img_feat).view(-1,self.joint_num,self.cfg.output_hm_shape[0],self.cfg.output_hm_shape[1],self.cfg.output_hm_shape[2])
        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord


class PositionMultiscaleNet(nn.Module):
    def __init__(self, cfg, channels):
        super(PositionMultiscaleNet, self).__init__()
        self.cfg = cfg
        in_channel = channels[-1]
        in_channel2 = channels[-2]
        if cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        kernel_size = getIntFromCfg(cfg, 'model_kernel_size', 1)
        if kernel_size == 1:
            padding = 0
        elif kernel_size == 3:
            padding = 1

        if getBooleanFromCfg(cfg, 'model_position_use_dw'):
            self.conv = make_dwconv_layers([in_channel,self.joint_num*self.cfg.output_hm_shape[0]], kernel=3, stride=1, padding=1, bnrelu_final=True)
            self.convh = make_dwconv_layers([in_channel2,self.joint_num*self.cfg.output_hm_shape[0]], kernel=3, stride=1, padding=1, bnrelu_final=True)
        else:
            self.conv = make_conv_layers([in_channel,self.joint_num*self.cfg.output_hm_shape[0]], kernel=kernel_size, stride=1, padding=padding, bnrelu_final=True)
            self.convh = make_conv_layers([in_channel2,self.joint_num*self.cfg.output_hm_shape[0]], kernel=kernel_size, stride=1, padding=padding, bnrelu_final=True)
        self.pool = nn.AdaptiveAvgPool2d((self.cfg.output_hm_shape[1],self.cfg.output_hm_shape[2]))
        final_conv_kernel_size = getIntFromCfg(cfg, 'model_final_conv_kernel_size', 1)
        if final_conv_kernel_size == 1:
            final_conv_padding = 0
        elif final_conv_kernel_size == 3:
            final_conv_padding = 1

        self.final_conv = make_conv_layers([2*self.joint_num*self.cfg.output_hm_shape[0], self.joint_num*self.cfg.output_hm_shape[0]], kernel=final_conv_kernel_size, stride=1, padding=final_conv_padding, bnrelu_final=False)

    def forward(self, img_feats):
        img_feat = img_feats[-1]
        joint_hm = self.conv(img_feat)
        t = self.convh(img_feats[-2])
        joint_hm2 = self.pool(t)

        joints = torch.cat((joint_hm, joint_hm2), dim=1)
        joint_hm = self.final_conv(joints).view(-1,self.joint_num,self.cfg.output_hm_shape[0],self.cfg.output_hm_shape[1],self.cfg.output_hm_shape[2])

        joint_coord = soft_argmax_3d(joint_hm)
        return joint_coord



class RotationNet(nn.Module):
    def __init__(self, cfg, in_channel):
        super(RotationNet, self).__init__()
        self.cfg = cfg
        if self.cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif self.cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        self.channel = getIntFromCfg(cfg, 'model_rotation_channel', 512)
       
        # output layers
        if self.cfg.parts == 'body':
            self.conv = make_conv_layers([in_channel,self.channel], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(self.channel+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(self.channel+3), (smpl.orig_joint_num-3)*6], relu_final=False) # without root and two hands
            self.shape_out = make_linear_layers([in_channel,smpl.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([in_channel,3], relu_final=False)
        elif self.cfg.parts == 'hand':
            self.conv = make_conv_layers([in_channel,self.channel], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(self.channel+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(self.channel+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
            self.shape_out = make_linear_layers([in_channel,mano.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([in_channel,3], relu_final=False)

    def forward(self, img_feat, joint_coord_img):
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        img_feat = self.conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)
        if self.cfg.parts == 'body':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        elif self.cfg.parts == 'hand':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param


class RotationMultiscaleNet(nn.Module):
    def __init__(self, cfg, channels):
        super(RotationMultiscaleNet, self).__init__()
        self.cfg = cfg
        in_channel = channels[-1]
        if self.cfg.parts == 'body':
            self.joint_num = smpl.pos_joint_num
        elif self.cfg.parts == 'hand':
            self.joint_num = mano.joint_num
        self.channel = getIntFromCfg(cfg, 'model_rotation_channel', 512)
       
        # output layers
        if self.cfg.parts == 'body':
            self.conv = make_conv_layers([in_channel,self.channel], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(self.channel+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(self.channel+3), (smpl.orig_joint_num-3)*6], relu_final=False) # without root and two hands
            self.shape_out = make_linear_layers([in_channel,smpl.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([in_channel,3], relu_final=False)
        elif self.cfg.parts == 'hand':
            self.conv = make_conv_layers([in_channel,self.channel], kernel=1, stride=1, padding=0)
            self.root_pose_out = make_linear_layers([self.joint_num*(self.channel+3), 6], relu_final=False)
            self.pose_out = make_linear_layers([self.joint_num*(self.channel+3), (mano.orig_joint_num-1)*6], relu_final=False) # without root joint
            self.shape_out = make_linear_layers([in_channel,mano.shape_param_dim], relu_final=False)
            self.cam_out = make_linear_layers([in_channel,3], relu_final=False)

    def forward(self, img_feats, joint_coord_img):
        img_feat = img_feats[-1]
        batch_size = img_feat.shape[0]

        # shape parameter
        shape_param = self.shape_out(img_feat.mean((2,3)))

        # camera parameter
        cam_param = self.cam_out(img_feat.mean((2,3)))
        
        # pose parameter
        img_feat = self.conv(img_feat)
        img_feat_joints = sample_joint_features(img_feat, joint_coord_img)
        feat = torch.cat((img_feat_joints, joint_coord_img),2)
        if self.cfg.parts == 'body':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        elif self.cfg.parts == 'hand':
            root_pose = self.root_pose_out(feat.view(batch_size,-1))
            pose_param = self.pose_out(feat.view(batch_size,-1))
        
        return root_pose, pose_param, shape_param, cam_param


class FaceRegressor(nn.Module):
    def __init__(self):
        super(FaceRegressor, self).__init__()
        self.pose_out = make_linear_layers([2048,12], relu_final=False) # pose parameter
        self.shape_out = make_linear_layers([2048, flame.shape_param_dim], relu_final=False) # shape parameter
        self.expr_out = make_linear_layers([2048, flame.expr_code_dim], relu_final=False) # expression parameter
        self.cam_out = make_linear_layers([2048,3], relu_final=False) # camera parameter

    def forward(self, img_feat):
        feat = img_feat.mean((2,3))
        
        # pose parameter
        pose_param = self.pose_out(feat)
        root_pose = pose_param[:,:6]
        jaw_pose = pose_param[:,6:]

        # shape parameter
        shape_param = self.shape_out(feat)

        # expression parameter
        expr_param = self.expr_out(feat)

        # camera parameter
        cam_param = self.cam_out(feat)

        return root_pose, jaw_pose, shape_param, expr_param, cam_param

