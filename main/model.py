import math
import copy
import os.path as osp
import glob

import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.resnet import ResNetBackbone
from torchvision.transforms import functional  as TVF
import numpy as np

from nets.module import PositionMultiscaleNet, PositionNet, RotationMultiscaleNet, RotationNet, FaceRegressor
from nets.loss import CoordLoss, ParamLoss
from utils.cfg_utils import getAnyFromCfg, getBooleanFromCfg, getFloatFromCfg, getStringFromCfg
from utils.debug import isDebug, printDebug
from utils.human_models import smpl, mano, flame
from utils.smplx.smplx.utils import SMPLOutput
from utils.transforms import batch_transform_joint_to_other_db, rot6d_to_axis_angle, transform_joint_to_other_db
#from config import cfg



class Model(nn.Module):
    def __init__(self, cfg, mode, networks):
        super(Model, self).__init__()
        self.cfg = cfg
        # body networks
        if cfg.parts == 'body':
            self.backbone = networks['backbone']
            self.position_net = networks['position_net']
            self.rotation_net = networks['rotation_net']
            self.smpl_layer = copy.deepcopy(smpl.layer['neutral'])

            if torch.cuda.is_available():
                self.smpl_layer = self.smpl_layer.cuda()

            self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]

        # hand networks
        elif cfg.parts == 'hand':
            self.backbone = networks['backbone']
            self.position_net = networks['position_net']
            self.rotation_net = networks['rotation_net']
            self.mano_layer = copy.deepcopy(mano.layer['right']).cuda()
            self.trainable_modules = [self.backbone, self.position_net, self.rotation_net]

        # face networks
        elif cfg.parts == 'face':
            self.backbone = networks['backbone']
            self.regressor = networks['regressor']
            self.flame_layer = copy.deepcopy(flame.layer).cuda()
            self.trainable_modules = [self.backbone, self.regressor]

        if mode == "train":
            self.coord_loss = CoordLoss()
            self.param_loss = ParamLoss()


    def get_camera_trans(self, cam_param):
        # camera translation
        t_xy = cam_param[:,:2]
        gamma = torch.sigmoid(cam_param[:,2]) # apply sigmoid to make it positive
        device = cam_param.device
        k_value = torch.FloatTensor([math.sqrt(self.cfg.focal[0]*self.cfg.focal[1]*self.cfg.camera_3d_size*self.cfg.camera_3d_size/(self.cfg.input_img_shape[0]*self.cfg.input_img_shape[1]))]).to(device).view(-1)

        t_z = k_value * gamma
        cam_trans = torch.cat((t_xy, t_z[:,None]),1)
        return cam_trans

    def forward_position_net(self, inputs, backbone, position_net):
        
        image = inputs['img']
        if getBooleanFromCfg(self.cfg, 'imagenet_normalize', False):
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            #image = (image - mean) / std
            image = TVF.normalize(image, mean, std, False)

        img_feat = backbone(image)
        joint_img = position_net(img_feat)
        return img_feat, joint_img
    
    def forward_rotation_net(self, img_feat, joint_img, rotation_net):
        if isinstance(img_feat, list):
            batch_size = img_feat[0].shape[0]
            device = img_feat[0].device
        else:
            batch_size = img_feat.shape[0]
            device = img_feat.device

        # parameter estimation
        if self.cfg.parts == 'body':
            root_pose_6d, pose_param_6d, shape_param, cam_param = rotation_net(img_feat, joint_img)
            # change 6d pose -> axis angles
            root_pose = rot6d_to_axis_angle(root_pose_6d)
            pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(batch_size,-1)
            zz = torch.zeros((batch_size,2*3), device=device).float()
            pose_param = torch.cat((pose_param, zz),1) # add two zero hand poses
            cam_trans = self.get_camera_trans(cam_param)
            return root_pose, pose_param, shape_param, cam_trans

        elif self.cfg.parts == 'hand':
            root_pose_6d, pose_param_6d, shape_param, cam_param = rotation_net(img_feat, joint_img)
            # change 6d pose -> axis angles
            root_pose = rot6d_to_axis_angle(root_pose_6d).reshape(-1,3)
            pose_param = rot6d_to_axis_angle(pose_param_6d.view(-1,6)).reshape(-1,(mano.orig_joint_num-1)*3)
            cam_trans = self.get_camera_trans(cam_param)
            return root_pose, pose_param, shape_param, cam_trans

    def get_coord(self, params, mode):
        batch_size = params['root_pose'].shape[0]
        device = params['root_pose'].device

        if self.cfg.parts == 'body':
            
            output = self.smpl_layer(global_orient=params['root_pose'], body_pose=params['body_pose'], betas=params['shape'])

            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(smpl.joint_regressor).to(device)[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = smpl.root_joint_idx
        elif self.cfg.parts == 'hand':
            output = self.mano_layer(global_orient=params['root_pose'], hand_pose=params['hand_pose'], betas=params['shape'])
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = torch.bmm(torch.from_numpy(mano.joint_regressor).to(device)[None,:,:].repeat(batch_size,1,1), mesh_cam)
            root_joint_idx = mano.root_joint_idx
        elif self.cfg.parts == 'face':
            zero_pose = torch.zeros((1,3), device=device).float().repeat(batch_size,1) # zero pose for eyes and neck
            output = self.flame_layer(global_orient=params['root_pose'], jaw_pose=params['jaw_pose'], betas=params['shape'], expression=params['expr'], neck_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose)
            # camera-centered 3D coordinate
            mesh_cam = output.vertices
            joint_cam = output.joints
            root_joint_idx = flame.root_joint_idx
        
        if mode == 'test' and self.cfg.testset == 'AGORA': # use 45 joints for AGORA evaluation
            joint_cam = output.joints
        
        #print ('SMPL output joints', output.joints.shape, joint_cam.shape)

        # project 3D coordinates to 2D space
        cam_trans = params['cam_trans']
        if mode == 'train' and len(self.cfg.trainset_3d) == 1 and self.cfg.trainset_3d[0] == 'AGORA' and len(self.cfg.trainset_2d) == 0: # prevent gradients from backpropagating to SMPL/MANO/FLAME paraemter regression module
            x = (joint_cam[:,:,0].detach() + cam_trans[:,None,0]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * self.cfg.focal[0] + self.cfg.princpt[0]
            y = (joint_cam[:,:,1].detach() + cam_trans[:,None,1]) / (joint_cam[:,:,2].detach() + cam_trans[:,None,2] + 1e-4) * self.cfg.focal[1] + self.cfg.princpt[1]
        else:
            x = (joint_cam[:,:,0] + cam_trans[:,None,0]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.cfg.focal[0] + self.cfg.princpt[0]
            y = (joint_cam[:,:,1] + cam_trans[:,None,1]) / (joint_cam[:,:,2] + cam_trans[:,None,2] + 1e-4) * self.cfg.focal[1] + self.cfg.princpt[1]
        x = x / self.cfg.input_img_shape[1] * self.cfg.output_hm_shape[2]
        y = y / self.cfg.input_img_shape[0] * self.cfg.output_hm_shape[1]
        joint_proj = torch.stack((x,y),2)

        # root-relative 3D coordinates
        root_cam = joint_cam[:,root_joint_idx,None,:]
        joint_cam = joint_cam - root_cam

        # add camera translation for the rendering
        mesh_cam = mesh_cam + cam_trans[:,None,:]
        return joint_proj, joint_cam, mesh_cam

    def forward(self, inputs, targets, meta_info, mode = "test", extra_info = {}):
        # network forward and get outputs
        # body network
        if self.cfg.parts == 'body':
            img_feat, joint_img = self.forward_position_net(inputs, self.backbone, self.position_net)
            if type(joint_img) is tuple:
                joint_img, pred_heatmap = joint_img
            smpl_root_pose, smpl_body_pose, smpl_shape, cam_trans = self.forward_rotation_net(img_feat, joint_img.detach(), self.rotation_net)
            dic = {'root_pose': smpl_root_pose, 'body_pose': smpl_body_pose, 'shape': smpl_shape, 'cam_trans': cam_trans}
            joint_proj, joint_cam, mesh_cam = self.get_coord(dic, mode)
            smpl_body_pose = smpl_body_pose.view(-1,(smpl.orig_joint_num-1)*3)
            smpl_pose = torch.cat((smpl_root_pose, smpl_body_pose),1)

        # hand network
        elif self.cfg.parts == 'hand':
            img_feat, joint_img = self.forward_position_net(inputs, self.backbone, self.position_net)
            mano_root_pose, mano_hand_pose, mano_shape, cam_trans = self.forward_rotation_net(img_feat, joint_img.detach(), self.rotation_net)

            joint_proj, joint_cam, mesh_cam = self.get_coord({'root_pose': mano_root_pose, 'hand_pose': mano_hand_pose, 'shape': mano_shape, 'cam_trans': cam_trans}, mode)
            mano_hand_pose = mano_hand_pose.view(-1,(mano.orig_joint_num-1)*3)
            mano_pose = torch.cat((mano_root_pose, mano_hand_pose),1)

        # face network
        elif self.cfg.parts == 'face':
            img_feat = self.backbone(inputs['img'])
            flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, cam_param = self.regressor(img_feat)
            flame_root_pose = rot6d_to_axis_angle(flame_root_pose)
            flame_jaw_pose = rot6d_to_axis_angle(flame_jaw_pose)
            cam_trans = self.get_camera_trans(cam_param)
            joint_proj, joint_cam, mesh_cam = self.get_coord({'root_pose': flame_root_pose, 'jaw_pose': flame_jaw_pose, 'shape': flame_shape, 'expr': flame_expr, 'cam_trans': cam_trans}, mode)
        
        if mode == 'train':
            # loss functions
            loss = {}
            if self.cfg.parts == 'body':
                if isDebug():
                    printDebug("Shape Parameters:")
                    printDebug("  Joint: ", joint_img.shape)
                    printDebug("  GT Joint: ", targets['joint_img'].shape)
                    printDebug("  GT Joint (Reduced)): ", smpl.reduce_joint_set(targets['joint_img']).shape)
                    

                    printDebug("  SMPL Pose:", smpl_pose.shape)
                    printDebug("  GT SMPL Pose:", targets['smpl_pose'].shape, meta_info['smpl_pose_valid'].shape)
                    printDebug("  SMPL SHAPE:", smpl_shape.shape)
                    printDebug("  GT Pose:", targets['smpl_shape'].shape)

                loss['joint_img'] = self.coord_loss(joint_img, smpl.reduce_joint_set(targets['joint_img']), smpl.reduce_joint_set(meta_info['joint_trunc']), meta_info['is_3D'])
                loss['smpl_joint_img'] = self.coord_loss(joint_img, smpl.reduce_joint_set(targets['smpl_joint_img']), smpl.reduce_joint_set(meta_info['smpl_joint_trunc']))
                loss['smpl_pose'] = self.param_loss(smpl_pose, targets['smpl_pose'], meta_info['smpl_pose_valid'])
                loss['smpl_shape'] = self.param_loss(smpl_shape, targets['smpl_shape'], meta_info['smpl_shape_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['smpl_joint_cam'] = self.coord_loss(joint_cam, targets['smpl_joint_cam'], meta_info['smpl_joint_valid'])
            elif self.cfg.parts == 'hand':
                loss['joint_img'] = self.coord_loss(joint_img, targets['joint_img'], meta_info['joint_trunc'], meta_info['is_3D'])
                loss['mano_joint_img'] = self.coord_loss(joint_img, targets['mano_joint_img'], meta_info['mano_joint_trunc'])
                loss['mano_pose'] = self.param_loss(mano_pose, targets['mano_pose'], meta_info['mano_pose_valid'])
                loss['mano_shape'] = self.param_loss(mano_shape, targets['mano_shape'], meta_info['mano_shape_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['mano_joint_cam'] = self.coord_loss(joint_cam, targets['mano_joint_cam'], meta_info['mano_joint_valid'])

            elif self.cfg.parts == 'face':
                loss['flame_root_pose'] = self.param_loss(flame_root_pose, targets['flame_root_pose'], meta_info['flame_root_pose_valid'][:,None])
                loss['flame_jaw_pose'] = self.param_loss(flame_jaw_pose, targets['flame_jaw_pose'], meta_info['flame_jaw_pose_valid'][:,None])
                loss['flame_shape'] = self.param_loss(flame_shape, targets['flame_shape'], meta_info['flame_shape_valid'][:,None])
                loss['flame_expr'] = self.param_loss(flame_expr, targets['flame_expr'], meta_info['flame_expr_valid'][:,None])
                loss['joint_proj'] = self.coord_loss(joint_proj, targets['joint_img'][:,:,:2], meta_info['joint_trunc'])
                loss['joint_cam'] = self.coord_loss(joint_cam, targets['joint_cam'], meta_info['joint_valid'] * meta_info['is_3D'][:,None,None])
                loss['flame_joint_cam'] = self.coord_loss(joint_cam, targets['flame_joint_cam'], meta_info['flame_joint_valid'])

            return loss
        else:
            # test output
            out = {'cam_trans': cam_trans} 
            if self.cfg.parts == 'body':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img
                out['smpl_mesh_cam'] = mesh_cam
                out['smpl_joint_proj'] = joint_proj
                out['smpl_pose'] = smpl_pose
                out['smpl_shape'] = smpl_shape
                # For validation, return GT
                if 'smpl_mesh_cam' in targets:
                    out['smpl_mesh_cam_target'] = targets['smpl_mesh_cam']
                if 'joint_img' in targets:
                    out['GT_joint_img'] = targets['joint_img']
                if 'joint_valid' in meta_info:
                    out['GT_joint_valid'] = meta_info['joint_valid']
                if 'bb2img_trans' in meta_info:
                    out['bb2img_trans'] = meta_info['bb2img_trans']

            elif self.cfg.parts == 'hand':
                out['img'] = inputs['img']
                out['joint_img'] = joint_img 
                out['mano_mesh_cam'] = mesh_cam
                out['mano_pose'] = mano_pose
                out['mano_shape'] = mano_shape
                if 'mano_mesh_cam' in targets:
                    out['mano_mesh_cam_target'] = targets['mano_mesh_cam']
                if 'joint_img' in targets:
                    out['joint_img_target'] = targets['joint_img']
                if 'joint_valid' in meta_info:
                    out['joint_valid'] = meta_info['joint_valid']
                if 'bb2img_trans' in meta_info:
                    out['bb2img_trans'] = meta_info['bb2img_trans']
            elif self.cfg.parts == 'face':
                out['img'] = inputs['img']
                out['flame_joint_cam'] = joint_cam
                out['flame_mesh_cam'] = mesh_cam
                out['flame_root_pose'] = flame_root_pose
                out['flame_jaw_pose'] = flame_jaw_pose
                out['flame_shape'] = flame_shape
                out['flame_expr'] = flame_expr
            return out


class TimmBackbone(nn.Module):
    def __init__(self, cfg, name, need_multiscale_key="need_multiscale"):
        super(TimmBackbone, self).__init__()
        import timm
        self.cfg = cfg

        self.need_multiscale = getBooleanFromCfg(self.cfg, need_multiscale_key, False)

        self.model = timm.create_model(name, features_only=True, pretrained=True)
        self.last_channel = self.model.feature_info[-1]['num_chs']
        
        if self.need_multiscale:
            self.channels = self.model.feature_info.channels()
        

    def init_weights(self):
        pass

    def forward(self,x):
        out = self.model(x)
        if not self.need_multiscale:
            return out[-1]
        return out



def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


def create_backbone(cfg):
    if cfg.backbone == "resnet":
        backbone = ResNetBackbone(cfg.resnet_type)
    elif cfg.backbone.startswith('timm_'):
        backbone = TimmBackbone(cfg, cfg.backbone[5:])
    return backbone


def get_model(cfg, mode):
    if cfg.parts == 'body':
        # Building backbone
        backbone = create_backbone(cfg)
        # Building Position decoder
        position_network = getStringFromCfg(cfg, 'position_net') 
        if position_network == "PositionMultiscaleNet":
            position_net = PositionMultiscaleNet(cfg, backbone.channels)
        else:
            position_net = PositionNet(cfg, backbone.last_channel)

        # Building Rotation decoder
        rotation_network = getStringFromCfg(cfg, 'rotation_net') 
        if rotation_network == "RotationMultiscaleNet":
            rotation_net = RotationMultiscaleNet(cfg, backbone.channels)
        else:
            rotation_net = RotationNet(cfg, backbone.last_channel)


        if (mode == 'train' or mode == "summary") and cfg.use_pretrained_weight:
            backbone.init_weights()
            position_net.apply(init_weights)
            rotation_net.apply(init_weights)
        model = Model(cfg, mode,  {'backbone': backbone, 'position_net': position_net, 'rotation_net': rotation_net})
        return model

    if cfg.parts == 'hand':
        backbone = ResNetBackbone(cfg.resnet_type)
        
        position_net = PositionNet(backbone.last_channel)
        rotation_net = RotationNet(backbone.last_channel)
        if mode == 'train':
            backbone.init_weights()
            position_net.apply(init_weights)
            rotation_net.apply(init_weights)
        model = Model(cfg, {'backbone': backbone, 'position_net': position_net, 'rotation_net': rotation_net})
        return model

    if cfg.parts == 'face':
        backbone = ResNetBackbone(cfg.resnet_type)
        regressor = FaceRegressor()
        if mode == 'train':
            backbone.init_weights()
            regressor.apply(init_weights)
        model = Model(cfg, {'backbone': backbone, 'regressor': regressor})
        return model

