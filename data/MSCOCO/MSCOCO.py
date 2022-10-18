import os
import os.path as osp
import random
import numpy as np
import copy
import json
import cv2
import torch
from pycocotools.coco import COCO
from base_dataset import BaseDataset
from utils.cfg_utils import getIntFromCfg
from utils.human_models import smpl, mano, flame
from utils.metrics_utils import keypoint_pck_accuracy
from utils.preprocessing import get_aug_config, load_img, process_bbox, augmentation, process_db_coord, process_human_model_output
from utils.transforms import transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, render_mesh
import utils.dataset_utils as du
from logger import mainlogger

skeleton = [(0,1), (0,2), (1,3), (2, 4), (3,5), (4,6),
    (0,7), (7,8), (7,9), (8,10), (9,11), (10, 12), (11,13),
    (12,14), (12, 15), (12, 16), (12,17), # hand
    (13, 18), (13,19), (13 ,20), (13, 21), # hand
    (7, 22), (22, 23), (22, 24), (23, 25), (24, 26), # head
    (5, 27), (5, 28), (5, 29), (27, 28), # foot
    (6, 30), (6, 31), (6, 32), (30, 31) # foot
    ]

class MSCOCO(BaseDataset):
    def __init__(self, cfg, transform, data_split):
        mainlogger.warning ("Loading COCO dataset ...")
        super().__init__(cfg, transform, data_split)

        self.img_path = osp.join('..', 'data', 'MSCOCO', 'images')
        self.annot_path = osp.join('..', 'data', 'MSCOCO', 'annotations')

        # mscoco joint set
        self.joint_set = {'body': \
                            {'joint_num': 32, 
                            'joints_name': ('Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle', 'Pelvis', 'L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel', 'L_Index_1', 'L_Middle_1', 'L_Ring_1', 'L_Pinky_1', 'R_Index_1', 'R_Middle_1', 'R_Ring_1', 'R_Pinky_1'),
                            'flip_pairs': ( (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16) , (18, 21), (19, 22), (20, 23), (24, 28), (25, 29) ,(26, 30), (27, 31) )
                            },\
                    'hand': \
                            {'joint_num': 21,
                            'joints_name': ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinky_4'),
                            'flip_pairs': ()
                            },
                    'face': \
                            {
                            'joint_to_flame': (-1, -1, -1, -1, -1, # no joints for neck, backheads, eyeballs
                                            17, 18, 19, 20, 21, # right eyebrow
                                            22, 23, 24, 25, 26, # left eyebrow
                                            27, 28, 29, 30, # nose
                                            31, 32, 33, 34, 35, # below nose
                                            36, 37, 38, 39, 40, 41, # right eye
                                            42, 43, 44, 45, 46, 47, # left eye
                                            48, # right lip
                                            49, 50, 51, 52, 53, # top lip
                                            54, # left lip
                                            55, 56, 57, 58, 59, # down lip
                                            60, 61, 62, 63, 64, 65, 66, 67, # inside of lip
                                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 # face contour
                                            )
                            }
                        }
        self.datalist = np.array(self.load_data())
        mainlogger.warning ("[MSCOCO] " + str(self.__len__()) + " annotations")
    
    def add_joint(self, joint_coord, feet_joint_coord, ljoint_coord, rjoint_coord):
        # pelvis
        lhip_idx = self.joint_set['body']['joints_name'].index('L_Hip')
        rhip_idx = self.joint_set['body']['joints_name'].index('R_Hip')
        pelvis = (joint_coord[lhip_idx,:] + joint_coord[rhip_idx,:]) * 0.5
        pelvis[2] = joint_coord[lhip_idx,2] * joint_coord[rhip_idx,2] # joint_valid
        pelvis = pelvis.reshape(1,3)
        
        # feet
        lfoot = feet_joint_coord[:3,:]
        rfoot = feet_joint_coord[3:,:]
        
        # hands
        lhand = ljoint_coord[[5,9,13,17], :]
        rhand = rjoint_coord[[5,9,13,17], :]

        joint_coord = np.concatenate((joint_coord, pelvis, lfoot, rfoot, lhand, rhand)).astype(np.float32)
        return joint_coord

    def load_data(self):
        if self.is_train():
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_train_v1.0.json'))
            if self.cfg.parts == 'body':
                with open(osp.join(self.annot_path, 'MSCOCO_train_SMPL_NeuralAnnot.json')) as f:
                    smpl_params = json.load(f)
            if self.cfg.parts == 'hand':
                with open(osp.join(self.annot_path, 'MSCOCO_train_MANO_NeuralAnnot.json')) as f:
                    mano_params = json.load(f)
            if self.cfg.parts == 'face':
                with open(osp.join(self.annot_path, 'MSCOCO_train_FLAME_NeuralAnnot.json')) as f:
                    flame_params = json.load(f)
        else:
            db = COCO(osp.join(self.annot_path, 'coco_wholebody_val_v1.0.json'))


        # train mode
        if self.is_train():
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('train2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                # body part
                if self.cfg.parts == 'body':
                    if ann['iscrowd'] or (ann['num_keypoints'] == 0):
                        continue
                    
                    # bbox
                    bbox = process_bbox(self.cfg, ann['bbox'], img['width'], img['height']) 
                    if bbox is None: continue
                    
                    # joint coordinates
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                    foot_joint_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1,3)
                    ljoint_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3)
                    rjoint_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)
                    joint_img = self.add_joint(joint_img, foot_joint_img, ljoint_img, rjoint_img)
                    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                    joint_img[:,2] = 0

                    said = str(aid)
                    smpl_param = smpl_params[said]

                    data_dict = {'key': aid, 'img_path': img_path, 
                        'img_shape': (img['height'],img['width']), 
                        'bbox': bbox, 
                        'joint_img': joint_img, 
                        'joint_valid': joint_valid, 
                        'smpl_param': smpl_param} 
                    datalist.append(data_dict)

                # hand part
                elif self.cfg.parts == 'hand':
                    
                    for hand_type in ('left', 'right'):
                        if ann[hand_type + 'hand_valid'] is False:
                            continue

                        bbox = process_bbox(self.cfg, ann[hand_type + 'hand_box'], img['width'], img['height'])
                        if bbox is None:
                            continue
                       
                        joint_img = np.array(ann[hand_type + 'hand_kpts'], dtype=np.float32).reshape(-1,3)
                        joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                        joint_img[:,2] = 0
     
                        mano_param = mano_params[str(aid)][hand_type]
                        if mano_param is not None:
                            mano_param['mano_param']['hand_type'] = hand_type
 
                        data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'mano_param': mano_param, 'hand_type': hand_type}
                        datalist.append(data_dict)

                # face part
                elif self.cfg.parts == 'face':

                    if ann['face_valid'] is False:
                        continue

                    bbox = process_bbox(self.cfg, ann['face_box'], img['width'], img['height'])
                    if bbox is None:
                        continue
 
                    joint_img = np.array(ann['face_kpts'], dtype=np.float32).reshape(-1,3)
                    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                    joint_img[:,2] = 0

                    # change keypoint set to that of flame
                    flame_joint_img = np.zeros((flame.joint_num,3), dtype=np.float32)
                    flame_joint_valid = np.zeros((flame.joint_num,1), dtype=np.float32)
                    for j in range(flame.joint_num):
                        if self.joint_set['face']['joint_to_flame'][j] == -1:
                            continue
                        flame_joint_img[j] = joint_img[self.joint_set['face']['joint_to_flame'][j]]
                        flame_joint_valid[j] = joint_valid[self.joint_set['face']['joint_to_flame'][j]]
                    joint_img = flame_joint_img
                    joint_valid = flame_joint_valid

                    flame_param = flame_params[str(aid)]
 
                    data_dict = {'img_path': img_path, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid, 'flame_param': flame_param}
                    datalist.append(data_dict)
            return datalist

        # test mode
        else:
            datalist = []
            for aid in db.anns.keys():
                ann = db.anns[aid]
                img = db.loadImgs(ann['image_id'])[0]
                imgname = osp.join('val2017', img['file_name'])
                img_path = osp.join(self.img_path, imgname)

                if self.cfg.parts == 'body':
                    bbox = process_bbox(self.cfg, ann['bbox'], img['width'], img['height']) 
                    if bbox is None: continue
                    
                    #joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                    joint_img = np.array(ann['keypoints'], dtype=np.float32).reshape(-1,3)
                    foot_joint_img = np.array(ann['foot_kpts'], dtype=np.float32).reshape(-1,3)
                    ljoint_img = np.array(ann['lefthand_kpts'], dtype=np.float32).reshape(-1,3)
                    rjoint_img = np.array(ann['righthand_kpts'], dtype=np.float32).reshape(-1,3)
                    joint_img = self.add_joint(joint_img, foot_joint_img, ljoint_img, rjoint_img)
                    joint_valid = (joint_img[:,2].copy().reshape(-1,1) > 0).astype(np.float32)
                    joint_img[:,2] = 0

                    data_dict = {'key': aid, 'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'joint_img': joint_img, 'joint_valid': joint_valid}
                    datalist.append(data_dict)

                elif self.cfg.parts == 'hand':
                    for hand_type in ('right','left'):
                        bbox = ann[hand_type + 'hand_box']
                        bbox = process_bbox(self.cfg, bbox, img['width'], img['height'])
                        if bbox is None:
                            continue
                        data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox, 'hand_type': hand_type}
                        datalist.append(data_dict)

                elif self.cfg.parts == 'face':
                    bbox = ann['face_box']
                    bbox = process_bbox(self.cfg, bbox, img['width'], img['height'])
                    if bbox is None:
                        continue

                    data_dict = {'img_path': img_path, 'ann_id': aid, 'img_shape': (img['height'],img['width']), 'bbox': bbox}
                    datalist.append(data_dict)
            return datalist
 
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])

        # train mode
        if self.data_split == 'train':
            img_path, img_shape = data['img_path'], data['img_shape']
            
            
            # image load
            img = load_img(img_path)

            # body part
            if self.cfg.parts == 'body':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.
    
                # coco gt
                dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(self.cfg, data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)

                # smpl fitted data
                smpl_param = data['smpl_param']
                smpl_joint_img, smpl_joint_cam, smpl_joint_trunc, smpl_pose, smpl_shape, smpl_mesh_cam_orig = process_human_model_output(self.cfg, smpl_param['smpl_param'], smpl_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'smpl')
                smpl_joint_valid = np.ones((smpl.joint_num,1), dtype=np.float32)
                smpl_pose_valid = np.ones((smpl.orig_joint_num*3), dtype=np.float32)
                smpl_shape_valid = float(True)

                """
                # for debug
                _tmp = smpl_joint_img.copy() 
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'smpl_joint_img': smpl_joint_img, 'smpl_joint_cam': smpl_joint_cam, 'smpl_pose': smpl_pose, 'smpl_shape': smpl_shape}
                meta_info = {'mesh' : smpl_mesh_cam_orig, 'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'smpl_joint_trunc': smpl_joint_trunc, 'smpl_joint_valid': smpl_joint_valid, 'smpl_pose_valid': smpl_pose_valid, 'smpl_shape_valid': smpl_shape_valid, 'is_3D': float(False)}
                return inputs, targets, meta_info

            # hand part
            elif self.cfg.parts == 'hand':
                # affine transform
                bbox, hand_type = data['bbox'], data['hand_type']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split, enforce_flip=(hand_type=='left')) # enforce flip when left hand to make it right hand
                img = self.transform(img.astype(np.float32))/255.
            
                # coco gt
                dummy_coord = np.zeros((self.joint_set['hand']['joint_num'],3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(self.cfg, data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['hand']['flip_pairs'], img2bb_trans, rot, self.joint_set['hand']['joints_name'], mano.joints_name)

                # mano fitted data
                mano_param = data['mano_param']
                mano_joint_img, mano_joint_cam, mano_joint_trunc, mano_pose, mano_shape, mano_mesh_cam_orig = process_human_model_output(self.cfg, mano_param['mano_param'], mano_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'mano')
                mano_joint_valid = np.ones((mano.joint_num,1), dtype=np.float32)
                mano_pose_valid = np.ones((mano.orig_joint_num*3), dtype=np.float32)
                mano_shape_valid = float(True)

                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '_' + hand_type + '.jpg', _img)
                _tmp = mano_joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '_' + hand_type + '_mano.jpg', _img)
                """

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'mano_joint_img': mano_joint_img, 'joint_cam': joint_cam, 'mano_joint_cam': mano_joint_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'mano_joint_trunc': mano_joint_trunc, 'mano_joint_valid': mano_joint_valid, 'mano_pose_valid': mano_pose_valid, 'mano_shape_valid': mano_shape_valid, 'is_3D': float(False)}
                return inputs, targets, meta_info

            # face part
            elif self.cfg.parts == 'face':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.

                # coco gt
                dummy_coord = np.zeros((flame.joint_num,3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(self.cfg, data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, flame.flip_pairs, img2bb_trans, rot, flame.joints_name, flame.joints_name)

                # flame fitted data
                flame_param = data['flame_param']
                flame_joint_img, flame_joint_cam, flame_joint_trunc, flame_root_pose, flame_jaw_pose, flame_shape, flame_expr, flame_joint_cam_orig, flame_mesh_cam_orig = process_human_model_output(self.cfg, flame_param['flame_param'], flame_param['cam_param'], do_flip, img_shape, img2bb_trans, rot, 'flame')
                flame_joint_valid = np.ones((flame.joint_num,1), dtype=np.float32)
                flame_root_pose_valid = float(True)
                flame_jaw_pose_valid = float(True)
                flame_shape_valid = float(True)
                flame_expr_valid = float(True)

                """
                # for debug
                #_tmp = flame_joint_img.copy()
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('coco_' + str(idx) + '.jpg', _img)
                """

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'joint_cam': joint_cam, 'flame_joint_cam': flame_joint_cam, 'flame_root_pose': flame_root_pose, 'flame_jaw_pose': flame_jaw_pose, 'flame_shape': flame_shape, 'flame_expr': flame_expr}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 'flame_joint_valid': flame_joint_valid, 'flame_root_pose_valid': flame_root_pose_valid, 'flame_jaw_pose_valid': flame_jaw_pose_valid, 'flame_shape_valid': flame_shape_valid, 'flame_expr_valid': flame_expr_valid, 'is_3D': float(False)}
                return inputs, targets, meta_info

        # test mode
        else:
            img_path, img_shape = data['img_path'], data['img_shape']

            # image load
            img = load_img(img_path)

            # body part
            if self.cfg.parts == 'body':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.
                
                dummy_coord = np.zeros((self.joint_set['body']['joint_num'],3), dtype=np.float32)
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(self.cfg, data['joint_img'], dummy_coord, data['joint_valid'], do_flip, img_shape, self.joint_set['body']['flip_pairs'], img2bb_trans, rot, self.joint_set['body']['joints_name'], smpl.joints_name)


                inputs = {'img': img}
                targets = {'joint_img' : joint_img}
                meta_info = {'bb2img_trans': bb2img_trans, 'joint_valid': joint_valid}
                return inputs, targets, meta_info

            # hand parts
            elif self.cfg.parts == 'hand':
                # affine transform
                bbox, hand_type = data['bbox'], data['hand_type']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split, enforce_flip=(hand_type=='left'))
                img = self.transform(img.astype(np.float32))/255.

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                return inputs, targets, meta_info

            # face parts
            elif self.cfg.parts == 'face':
                # affine transform
                bbox = data['bbox']
                img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(self.cfg, img, bbox, self.data_split)
                img = self.transform(img.astype(np.float32))/255.

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                return inputs, targets, meta_info
    
    def evaluate(self, outs, cur_sample_idx, viz=False):
        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'avg_pck': []}

        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            ann_id = annot['ann_id']
            out = outs[n]

            if self.cfg.parts == 'body':


                smpl_joint_img = out['smpl_joint_proj']
                gt_joint_img = out['GT_joint_img']
                gt_joint_valid = np.reshape(out['GT_joint_valid'], (1, smpl.joint_num)).astype(np.bool)

                if np.count_nonzero(gt_joint_valid) >= 1:
                    _tmp = smpl_joint_img.copy()
                    _tmp[:,0] = _tmp[:,0] / self.cfg.output_hm_shape[2] * self.cfg.input_img_shape[1]
                    _tmp[:,1] = _tmp[:,1] / self.cfg.output_hm_shape[1] * self.cfg.input_img_shape[0]

                    _tmp2 = gt_joint_img.copy()
                    _tmp2[:,0] = _tmp2[:,0] / self.cfg.output_hm_shape[2] * self.cfg.input_img_shape[1]
                    _tmp2[:,1] = _tmp2[:,1] / self.cfg.output_hm_shape[1] * self.cfg.input_img_shape[0]

                    gt_coco_keypoints = np.array([_tmp2.copy()[:,0:2]])
                    res_coco_keypoints = np.array([_tmp.copy()[:,0:2]])

                    normalize = np.array([[self.cfg.input_img_shape[1], self.cfg.input_img_shape[0]]])
                    pcks, avg_pck2, cnt = keypoint_pck_accuracy(res_coco_keypoints, gt_coco_keypoints, gt_joint_valid , 0.05, normalize )
                    
                    eval_result['avg_pck'].append(avg_pck2)
                    #print ('Average PCK for ', ann_id, annot['img_path'], avg_pck2)

            elif self.cfg.parts == 'hand':
                vis = False
                if vis:
                    #save_obj(out['mano_mesh_cam'], mano.face['right'], str(ann_id) + '_rhand.obj')
                    #save_obj(out['mano_mesh_cam'], mano.face['left'], str(ann_id) + '_lhand.obj')

                    # all hands are flipped to the right hand.
                    # restore to the left hand.
                    hand_type, bbox = annot['hand_type'], annot['bbox']
                    mano_pose = out['mano_pose']; mano_shape = out['mano_shape']; mano_trans = out['cam_trans']
                    if hand_type == 'left':
                        mano_pose = mano_pose.reshape(-1,3)
                        mano_pose[:,1:3] *= -1
                        mano_trans[0] *= -1

                    focal_x = self.cfg.focal[0] / self.cfg.input_img_shape[1] * bbox[2]
                    focal_y = self.cfg.focal[1] / self.cfg.input_img_shape[0] * bbox[3]
                    princpt_x = self.cfg.princpt[0] / self.cfg.input_img_shape[1] * bbox[2] + bbox[0]
                    princpt_y = self.cfg.princpt[1] / self.cfg.input_img_shape[0] * bbox[3] + bbox[1]
                    save_dict = {'mano_param': {'pose': mano_pose.reshape(-1).tolist(), 'shape': mano_shape.reshape(-1).tolist(), 'trans': mano_trans.reshape(-1).tolist()},\
                                'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                                }
                    with open(osp.join(self.cfg.result_dir, 'mano_param_' + hand_type + '_' + str(ann_id) + '.json'), 'w') as f:
                        json.dump(save_dict, f)

            
            elif self.cfg.parts == 'face':
                vis = False
                if vis:
                    #img = (out['img'].transpose(1,2,0)[:,:,::-1] * 255).copy()
                    #cv2.imwrite(str(ann_id) + '.jpg', img)
                    #save_obj(out['flame_mesh_cam'], flame.face, str(ann_id) + '_face.obj')

                    bbox = annot['bbox']
                    flame_root_pose = out['flame_root_pose']; flame_jaw_pose = out['flame_jaw_pose']; flame_shape = out['flame_shape']; flame_expr = out['flame_expr']; flame_trans = out['cam_trans']
                    focal_x = self.cfg.focal[0] / self.cfg.input_img_shape[1] * bbox[2]
                    focal_y = self.cfg.focal[1] / self.cfg.input_img_shape[0] * bbox[3]
                    princpt_x = self.cfg.princpt[0] / self.cfg.input_img_shape[1] * bbox[2] + bbox[0]
                    princpt_y = self.cfg.princpt[1] / self.cfg.input_img_shape[0] * bbox[3] + bbox[1]
                    save_dict = {'flame_param': {'root_pose': flame_root_pose.reshape(-1).tolist(), 'jaw_pose': flame_jaw_pose.reshape(-1).tolist(), 'shape': flame_shape.reshape(-1).tolist(), 'expr': flame_expr.reshape(-1).tolist(), 'trans': flame_trans.reshape(-1).tolist()},\
                                'cam_param': {'focal': (focal_x,focal_y), 'princpt': (princpt_x,princpt_y)}
                                }
                    with open(osp.join(self.cfg.result_dir, 'flame_param_' + str(ann_id) + '.json'), 'w') as f:
                        json.dump(save_dict, f)


        return eval_result

    def print_eval_result(self, eval_result):
        return du.print_eval_result(eval_result)

