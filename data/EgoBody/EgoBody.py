import ast
import sys
from audioop import add
from logging import root
import os
import os.path as osp
import random
import numpy as np
import copy
import json
import cv2
import torch
import pandas as pd
import pickle
import math
import torch.nn as nn
from tqdm import tqdm

from pycocotools.coco import COCO
from base_dataset import BaseDataset
from utils.cfg_utils import getBooleanFromCfg, getStringFromCfg
from utils.dir import make_folder
from utils.human_models import smpl
from utils.json_utils import load_json, write_json
from utils.metrics_utils import keypoint_mpjpe
from utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import rigid_align, transform_joint_to_other_db
from utils.vis import save_obj, render_mesh, vis_keypoints
import utils.dataset_utils as du
from logger import mainlogger
import glob
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import torchgeometry as tgm


ego_body_device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")


SMPL_JOINT_VALID = np.ones((smpl.joint_num,1), dtype=np.float32)
SMPL_POSE_VALID = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
SMPL_SHAPE_VALID = float(True)

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord

# From here:
# https://github.com/sanweiliti/PLACE/blob/ce49f99ca5a8b62366def9dcb63edf14ec8493d2/utils.py
def update_globalRT_for_smpl(smpl_param, smpl, M, delta_T = None):
        # smpl fitted data
        human_model_param = {}

        body_params_dict = {}
        body_params_dict['transl'] = smpl_param['trans']
        body_params_dict['global_orient'] = smpl_param['global_orient']
        body_params_dict['betas'] = smpl_param['shape']
        body_params_dict['body_pose'] = smpl_param['body_pose']

        body_param_dict_torch = {}
        for key in body_params_dict.keys():
            body_param_dict_torch[key] = torch.FloatTensor(body_params_dict[key]).to(ego_body_device)

        #delta_T = np.zeros((3, 1))
        if delta_T is None:
            with torch.no_grad():
                body_param_dict_torch['transl'] = torch.zeros([1,3], dtype=torch.float32).to(ego_body_device)
                body_param_dict_torch['global_orient'] = torch.zeros([1,3], dtype=torch.float32).to(ego_body_device)
                output = smpl(return_verts=True, **body_param_dict_torch)

                delta_T = output.joints[0,0,:] # (3,)
                delta_T = delta_T.detach().cpu().numpy()

        body_R_angle = body_params_dict['global_orient'][0]
        body_R_mat = R.from_rotvec(body_R_angle).as_matrix() # to a [3,3] rotation mat
        body_T = body_params_dict['transl'][0]
        body_mat = np.eye(4)
        body_mat[:-1,:-1] = body_R_mat
        body_mat[:-1, -1] = body_T + delta_T

        ### step (3): perform transformation, and decalib the delta shift
        body_params_dict_new = copy.deepcopy(body_params_dict)
        body_mat_new = np.dot(M, body_mat)
        body_R_new = R.from_matrix(body_mat_new[:-1,:-1]).as_rotvec()
        body_T_new = body_mat_new[:-1, -1]
        body_params_dict_new['global_orient'] = body_R_new.reshape(1,3)
        body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(1,3)

        human_model_param['trans'] = body_params_dict_new['transl']
        human_model_param['shape'] = body_params_dict_new['betas']
        root_pose = body_params_dict_new['global_orient']
        human_model_param['global_orient'] = root_pose
        human_model_param['body_pose'] = body_params_dict_new['body_pose']

        # body_param_new = body_params_parse(body_params_dict_new)[0]  # array, [72]

        human_model_param['pose'] = np.concatenate((root_pose, human_model_param['body_pose']), axis=1)

        return human_model_param



class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class EgoBody(BaseDataset):
    def __init__(self, cfg, transform, data_split, visualize_info=False):
        mainlogger.warning ("Loading EgoBody dataset ...")
        super().__init__(cfg, transform, data_split)

        self.visualize_info = visualize_info
        self.data_path = osp.join('..', 'data', 'EgoBody', 'data')
        self.img_path = osp.join('..', 'data', 'EgoBody', 'data')


        self.joint_set = {'body': {'joint_num': 25, 
                            # OpenPose Body 25
                            'joints_name': ('Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist',  #7
                            'Pelvis', 'R_Hip',  'R_Knee', 'R_Ankle', 'L_Hip',  'L_Knee', 'L_Ankle',   #14
                            'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear',  #18
                            'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'  #24
                            ),
                            'flip_pairs': ( (2, 5), (3, 6), (4, 7), (9, 12), (10, 13), (11, 14), (15, 16), (17, 18), (19, 22), (20, 23), (21, 24)),
                            'eval_joint': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 , 24),
                            }
            }
        self.joint_set['body']['root_joint_idx'] = self.joint_set['body']['joints_name'].index('Pelvis')


        self.datalist = np.array(self.load_data())
        mainlogger.warning ("[EgoBody(" + self.data_split + ")] " + str(self.__len__()) + " annotations")

    def load_data(self):
        
        json_cache = load_json(osp.join(self.data_path, "cache.json"), default={})
        save_cache = True
        if len(json_cache.keys()) > 1:
            mainlogger.warning ("Using JSON Cache for EgoBody")

        df = pd.read_csv(os.path.join(self.data_path, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        start_frame_list = list(df['start_frame'])
        end_frame_list = list(df['end_frame'])
        body_idx_fpv_list = list(df['body_idx_fpv'])

        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
        start_frame_dict = dict(zip(recording_name_list, start_frame_list))
        end_frame_dict = dict(zip(recording_name_list, end_frame_list))
        # get body idx, gender for the interactee
        

        df2 = pd.read_csv(os.path.join(self.data_path, 'data_splits.csv'))
        if getBooleanFromCfg(self.cfg, "dataset_use_val_egobody", False):
            if self.is_train():
                recording_names = list(df2["train"]) + list(df2["val"])
            else:
                recording_names = list(df2[self.data_split])
        else:
            recording_names = list(df2[self.data_split])

        #print (recording_names)
        datalist = []

        for recording_name in tqdm(recording_names):
            if osp.exists(osp.join(self.data_path, 'egocentric_color', str(recording_name))):
                calib_trans_dir = os.path.join(self.data_path, 'calibrations', recording_name)  # extrinsics
                fpv_recording_dir = glob.glob(os.path.join(self.data_path, 'egocentric_color', recording_name, '202*'))[0]
                
                interactee_idx = int(body_idx_fpv_dict[recording_name].split(' ')[0])
                interactee_gender = body_idx_fpv_dict[recording_name].split(' ')[1]
                fitting_root_interactee = osp.join(self.data_path, 'smpl_interactee', recording_name)

                holo_pv_path_list = glob.glob(os.path.join(fpv_recording_dir, 'PV', '*_frame_*.jpg'))
                holo_pv_path_list = sorted(holo_pv_path_list)
                holo_frame_id_list = [osp.basename(x).split('.')[0].split('_', 1)[1] for x in holo_pv_path_list]
                holo_frame_id_dict = dict(zip(holo_frame_id_list, holo_pv_path_list))
                holo_timestamp_list = [osp.basename(x).split('_')[0] for x in holo_pv_path_list]
                holo_timestamp_dict = dict(zip(holo_timestamp_list, holo_frame_id_list))
                #print ('holo_timestamp_dict', holo_timestamp_dict)
                #print ('holo_frame_id_dict', holo_frame_id_dict)

                valid_frame_npz = osp.join(fpv_recording_dir, 'valid_frame.npz')
                kp_npz = osp.join(fpv_recording_dir, 'keypoints.npz')
                valid_frames = np.load(valid_frame_npz)
                holo_2djoints_info = np.load(kp_npz)
                assert len(valid_frames['valid']) == len(valid_frames['imgname'])

                holo_frame_id_all = [osp.basename(x).split('.')[0].split('_', 1)[1] for x in valid_frames['imgname']]
                holo_valid_dict = dict(zip(holo_frame_id_all, valid_frames['valid']))  # 'frame_01888': True

                holo_frame_id_valid = [osp.basename(x).split('.')[0].split('_', 1)[1] for x in holo_2djoints_info['imgname']]  # list of all valid frame names (e.x., 'frame_01888')
                holo_keypoint_dict = dict(zip(holo_frame_id_valid, holo_2djoints_info['keypoints']))
                holo_center_dict = dict(zip(holo_frame_id_valid, holo_2djoints_info['center']))
                holo_scale_dict = dict(zip(holo_frame_id_valid, holo_2djoints_info['scale']))


                ################################## read hololens world <-> kinect master RGB cam extrinsics
                holo2kinect_dir = os.path.join(calib_trans_dir, 'cal_trans', 'holo_to_kinect12.json')
                with open(holo2kinect_dir, 'r') as f:
                    trans_holo2kinect = np.array(json.load(f)['trans'])
                trans_kinect2holo = np.linalg.inv(trans_holo2kinect)


                ######## read hololens camera info
                # for each sequence: unique cx, cy, w, h
                # for each frame: different fx, fy, pv2world_transform
                pv_info_path = glob.glob(os.path.join(fpv_recording_dir, '*_pv.txt'))[0]
                with open(pv_info_path) as f:
                    lines = f.readlines()
                holo_cx, holo_cy, holo_w, holo_h = ast.literal_eval(lines[0])  # hololens pv camera infomation

                holo_fx_dict = {}
                holo_fy_dict = {}
                holo_pv2world_trans_dict = {}
                for i, frame in enumerate(lines[1:]):
                    frame = frame.split((','))
                    cur_timestamp = frame[0]  # string
                    cur_fx = float(frame[1])
                    cur_fy = float(frame[2])
                    cur_pv2world_transform = np.array(frame[3:20]).astype(float).reshape((4, 4))

                    if cur_timestamp in holo_timestamp_dict.keys():
                        cur_frame_id = holo_timestamp_dict[cur_timestamp]
                        holo_fx_dict[cur_frame_id] = cur_fx
                        holo_fy_dict[cur_frame_id] = cur_fy
                        holo_pv2world_trans_dict[cur_frame_id] = cur_pv2world_transform

                smpl_js = {}
                if os.path.exists(os.path.join(fitting_root_interactee, 'smpl.json')):
                    with open(os.path.join(fitting_root_interactee, 'smpl.json'), 'r') as json_file:
                        smpl_js = json.load(json_file)

                for i_frame in range(start_frame_dict[recording_name], end_frame_dict[recording_name]+1):
                    holo_frame_id = 'frame_{}'.format("%05d" % i_frame)

                    interactee_exist_key = recording_name + '#' + str(interactee_idx) + '#' + str(holo_frame_id) + 'pkl'
                    if interactee_exist_key in json_cache:
                        has_pkl = json_cache[interactee_exist_key]
                    else:
                        has_pkl = osp.exists(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', holo_frame_id, '000.pkl'))
                        json_cache[interactee_exist_key] = has_pkl
                        #print('interactee fitting {} do not exist!'.format(holo_frame_id))
                    if not has_pkl:    
                        continue

                    if holo_frame_id not in holo_frame_id_dict.keys():  # the frame is dropped in hololens recording
                        pass
                    elif (holo_frame_id in holo_valid_dict.keys()) and holo_valid_dict[holo_frame_id]:
                        #print ('fpv_recording_dir', fpv_recording_dir)
                        #print ('holo_dict', holo_frame_id_dict[holo_frame_id])
                        fpv_img_path = holo_frame_id_dict[holo_frame_id]
                        keypoints_holo = holo_keypoint_dict[holo_frame_id]  # [25, 3] openpose detections

                        center = holo_center_dict[holo_frame_id]
                        scale = holo_scale_dict[holo_frame_id]

                        cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
                        cur_world2pv_transform = np.linalg.inv(cur_pv2world_transform)


                        cur_fx = holo_fx_dict[holo_frame_id] 
                        cur_fy = holo_fy_dict[holo_frame_id] 
                        focal = (cur_fx, cur_fy)
                        princpt = (holo_cx, holo_cy)


                        img_path = fpv_img_path
                        img = {'height' : 1080, 'width': 1920}
                        bbox = [0, 0, 1920, 1080]

                        # using center and scale
                        h = scale * 200
                        x1 = center[0] - h/2
                        y1 = center[1] - h/2
                        bbox = [x1, y1, h, h]
                        #print ('center', center, 'princpt', princpt)

                        joint_img = keypoints_holo
                        joint_valid =  np.expand_dims(np.array(keypoints_holo[:, 2] != 0), 1).astype(np.float)   #np.ones((25, 1))
                        for (i, k) in enumerate(keypoints_holo):
                            if k[0] == 0.0 and k[1] == 0.0:
                                joint_valid[i] = 0
                        smpl_param = {'cam_param': {'focal': focal, 'princpt': princpt}}

                        param_json_cache_file = osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', holo_frame_id, '000.json')
                        if smpl_js is not None and holo_frame_id in smpl_js:
                            human_model_param = smpl_js[holo_frame_id]
                            for k in ["pose", "trans", "shape", "global_orient", "body_pose"]:
                                human_model_param[k] = np.array(human_model_param[k])
                        elif osp.exists(param_json_cache_file):
                            with open(param_json_cache_file, 'r') as f:
                                human_model_param = json.load(f)
                            for k in ["pose", "trans", "shape", "global_orient", "body_pose"]:
                                human_model_param[k] = np.array(human_model_param[k])
                            smpl_js[holo_frame_id] = human_model_param
                        else:

                            with open(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', holo_frame_id, '000.pkl'), 'rb') as f:
                                param = pickle.load(f)
                            smpl_param['smpl_param'] = param

                            # smpl fitted data
                            #smpl_param = data['smpl_param']
                            human_model_param = {}
                            human_model_param['trans'] = smpl_param['smpl_param']['transl']
                            human_model_param['shape'] = smpl_param['smpl_param']['betas']
                            human_model_param['global_orient'] = smpl_param['smpl_param']['global_orient']
                            human_model_param['body_pose'] = smpl_param['smpl_param']['body_pose']
                            
                            add_trans = np.array([[1.0, 0, 0, 0],
                                                            [0, -1, 0, 0],
                                                            [0, 0, -1, 0],
                                                            [0, 0, 0, 1]])  # different y/z axis definition in opencv/opengl convention
                            
                            body_params_dict = {}
                            body_params_dict['transl'] = human_model_param['trans']
                            body_params_dict['global_orient'] = human_model_param['global_orient']
                            body_params_dict['betas'] = human_model_param['shape']
                            body_params_dict['body_pose'] = human_model_param['body_pose']

                            body_model = smpl.layer[interactee_gender].to(ego_body_device)

                            body_param_dict_torch = {}
                            for key in body_params_dict.keys():
                                body_param_dict_torch[key] = torch.tensor(body_params_dict[key], dtype=torch.float32, device=ego_body_device) #torch.FloatTensor(body_params_dict[key]).to(ego_body_device)

                            with torch.no_grad():
                                body_param_dict_torch['transl'] = torch.zeros([1,3], dtype=torch.float32, device=ego_body_device)
                                body_param_dict_torch['global_orient'] = torch.zeros([1,3], dtype=torch.float32, device=ego_body_device)
                                output = body_model(return_verts=True, **body_param_dict_torch)

                                delta_T = output.joints[0,0,:] # (3,)
                                delta_T = delta_T.detach().cpu().numpy()

                            human_model_param = update_globalRT_for_smpl(human_model_param, body_model, trans_kinect2holo, delta_T=delta_T)
                            human_model_param = update_globalRT_for_smpl(human_model_param, body_model, cur_world2pv_transform, delta_T=delta_T)
                            human_model_param = update_globalRT_for_smpl(human_model_param, body_model, add_trans, delta_T=delta_T)
                            
                            #root_pose = human_model_param['global_orient']
                            #human_model_param['pose'] = np.concatenate((root_pose, human_model_param['body_pose']), axis=1)
                            human_model_param['gender'] = interactee_gender
                            with open(param_json_cache_file, 'w') as json_file:
                                json.dump(human_model_param, json_file, cls=NumpyArrayEncoder)

                        data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'gender': interactee_gender,
                        'joint_valid': joint_valid, 'smpl_param': smpl_param,
                        'recording_name' : recording_name, 'frame_name': osp.basename(img_path), 'human_model_param': human_model_param
                        } 
                        datalist.append(data_dict)
        
                #with open(os.path.join(fitting_root_interactee, 'smpl.json'), 'w') as json_file:
                #    json.dump(smpl_js, json_file, cls=NumpyArrayEncoder)

        if save_cache:
            write_json(osp.join(self.data_path, "cache.json"), json_cache)
            
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        
        img_path, img_shape = data['img_path'], data['img_shape']

        # image load
        img = load_img(img_path)

        # affine transform
        bbox = data['bbox']
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32))/255.
        
        # coco gt
        dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
        joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(self.cfg, data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)


        smpl_param = data['smpl_param']
        cam_param = smpl_param['cam_param']
        human_model_param = data['human_model_param']

        smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(self.cfg, human_model_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smpl')
        
        smpl_joint_valid = SMPL_JOINT_VALID.copy()
        smpl_pose_valid = SMPL_POSE_VALID.copy()
        smpl_shape_valid = SMPL_SHAPE_VALID

        
        inputs = { 'img': img }
        targets = { 'joint_img': joint_img, 'joint_cam': joint_cam, 'smpl_joint_img': smpl_joint_img, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
        meta_info = { 'mesh' : smpl_mesh_cam_orig, 'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(False)}
        if self.visualize_info:
            debug = {"smpl_mesh_cam_orig": smpl_mesh_cam_orig, 'img_path': img_path, 'cam_param': data['smpl_param']['cam_param'], 'bbox': data['bbox'], 'ori_2djoints': data['joint_img'], 'ori_2djoints_valid': data['joint_valid']}
            meta_info['debug'] = debug
        if self.data_split == "val" or self.data_split == "test":
            meta_info['recording_name'] = data['recording_name']
            meta_info['frame_name'] = data['frame_name']
            targets['smpl_mesh_cam'] = smpl_mesh_cam_orig
        
        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx, viz=False):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': [], 'mpvpe': [], 'pa_mpvpe': []}

        for n in range(sample_num):
            out = outs[n]
   
            # EgoBody joint from gt mesh
            mesh_gt_cam = out['smpl_mesh_cam_target'].copy()
            pose_coord_gt_pw3d = np.dot(smpl.joint_regressor, mesh_gt_cam)
            pose_coord_gt_pw3d = pose_coord_gt_pw3d - pose_coord_gt_pw3d[self.joint_set['body']['root_joint_idx'],None] # root-relative
            mesh_gt_cam -= np.dot(smpl.joint_regressor, mesh_gt_cam)[0,None,:]
            
            # EgoBody joint from output mesh
            mesh_out_cam = out['smpl_mesh_cam'].copy()
            pose_coord_out_pw3d = np.dot(smpl.joint_regressor, mesh_out_cam)
            pose_coord_out_pw3d = pose_coord_out_pw3d - pose_coord_out_pw3d[self.joint_set['body']['root_joint_idx'],None] # root-relative

            pose_coord_out_pw3d_aligned = rigid_align(pose_coord_out_pw3d, pose_coord_gt_pw3d)
            eval_result['mpjpe'].append(np.sqrt(np.sum((pose_coord_out_pw3d - pose_coord_gt_pw3d)**2,1)).mean() * 1000) # meter -> milimeter
            eval_result['pa_mpjpe'].append(np.sqrt(np.sum((pose_coord_out_pw3d_aligned - pose_coord_gt_pw3d)**2,1)).mean() * 1000) # meter -> milimeter
            

            mesh_out_cam -= np.dot(smpl.joint_regressor, mesh_out_cam)[0,None,:]
            mesh_out_cam_aligned = rigid_align(mesh_out_cam, mesh_gt_cam)
            eval_result['mpvpe'].append(np.sqrt(np.sum((mesh_out_cam - mesh_gt_cam)**2,1)).mean() * 1000) # meter -> milimeter
            eval_result['pa_mpvpe'].append(np.sqrt(np.sum((mesh_out_cam_aligned - mesh_gt_cam)**2,1)).mean() * 1000) # meter -> milimeter

        return eval_result

    def print_eval_result(self, eval_result):
        return du.print_eval_result(eval_result)

