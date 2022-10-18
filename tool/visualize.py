"""
Tool to visualize dataset
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License

"""
import sys
import os
import os.path as osp
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import argparse
import json
import trimesh
from visualize_body_part_segmentation import binarysegm_to_vertex_colors
from visualize_body_part_segmentation import part_segm_to_vertex_colors


cur_dir = osp.dirname(os.path.abspath(__file__))
root_dir = osp.join(cur_dir, '..')
sys.path.insert(0, osp.join(root_dir, 'common'))
sys.path.insert(0, osp.join(root_dir, 'data'))
sys.path.insert(0, osp.join(root_dir, 'main'))

from config import cfg
from utils.vis import render_mesh, save_obj, vis_mesh, vis_keypoints, vis_keypoints_with_skeleton
from utils.human_models import smpl
from utils.preprocessing import load_img
from logger import mainlogger
from utils.dir import add_pypath

#from AGORA import AGORA
#from MPI_INF_3DHP import MPI_INF_3DHP
#from CrowdPose import CrowdPose
#from InstaVariety import InstaVariety
#from MSCOCO import MSCOCO
import pyrender


skeleton = [(0,1), (0,2), (1,3), (2, 4), (3,5), (4,6),
    (0,7), (7,8), (7,9), (8,10), (9,11), (10, 12), (11,13),
    (12,14), (12, 15), (12, 16), (12,17), # hand
    (13, 18), (13,19), (13 ,20), (13, 21), # hand
    (7, 22), (22, 23), (22, 24), (23, 25), (24, 26), # head
    (5, 27), (5, 28), (5, 29), (27, 28), # foot
    (6, 30), (6, 31), (6, 32), (30, 31) # foot
    ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default="EgoBody")
    parser.add_argument('--aug', type=int, default=-1)
    
    args = parser.parse_args()
    cfg.load(args.config)


    mainlogger.warning ("Visualizing datasets ....")

    cfg.bbox_3d_size = 2
    cfg.camera_3d_size = 2.5
    cfg.input_img_shape = (256, 192)
    cfg.output_hm_shape = (8, 8, 6)
    cfg.focal = (5000, 5000) # virtual focal lengths
    cfg.princpt = (cfg.input_img_shape[1]/2, cfg.input_img_shape[0]/2) # virtual principal point position
    cfg.augmentation = args.aug # -1  # No augmentation
    #cfg.use_occluder = True
    #cfg.occlusion_prob = 1.0

    add_pypath(osp.join(cfg.data_dir, args.dataset))

    if str(args.dataset).lower() == "MPI_INF_3DHP".lower():
        from MPI_INF_3DHP import MPI_INF_3DHP
        dataset = MPI_INF_3DHP(cfg, transforms.ToTensor(), "train", visualize_info=True)
    elif str(args.dataset).lower() == "CrowdPose".lower():
        from CrowdPose import CrowdPose
        dataset = CrowdPose(cfg, transforms.ToTensor(), "val", visualize_info=True)
    elif str(args.dataset).lower() == "InstaVariety".lower():
        from InstaVariety import InstaVariety
        dataset = InstaVariety(cfg, transforms.ToTensor(), "test", visualize_info=True)
    elif str(args.dataset).lower() == "EgoBody".lower():
        from EgoBody import EgoBody
        dataset = EgoBody(cfg, transforms.ToTensor(), "train", visualize_info=True)
    elif str(args.dataset).lower() == "Agora".lower():
        dataset = AGORA(cfg, transforms.ToTensor(), "train")
    elif str(args.dataset).lower() == "MSCOCO".lower():
        dataset = MSCOCO(cfg, transforms.ToTensor(), "val")
    elif str(args.dataset).lower() == "H36M".lower():
        from Human36M import Human36M
        dataset = Human36M(cfg, transforms.ToTensor(), "train")
    else:
        print ("Unknown datset ....")

    #for i in range(25):
    #    test_data_augm(dataset, 0, i)
    process(dataset, 0)
    process(dataset, 200)
    process(dataset, 1000)
    #process(dataset, 2000)
    #for i in range(10):
    #    process(dataset, i)


def check_gt_params(inputs, targets, meta_info):
    assert 'img' in inputs
    #print ('Img shape', inputs['img'].shape)
    #print ('Img dtype', inputs['img'].dtype)
    assert list(inputs['img'].shape) == [3, 256, 192]
    assert inputs['img'].dtype == torch.float32

    assert 'joint_img' in targets
    assert list(targets['joint_img'].shape) == [33,3]
    #print ('Joint type', targets['joint_img'].shape)

    assert 'smpl_joint_img' in targets
    assert list(targets['smpl_joint_img'].shape) == [33,3]
    #print ('smpl_joint_img', targets['smpl_joint_img'].shape)

    assert 'joint_cam' in targets
    assert list(targets['joint_cam'].shape) == [33,3]
    #print ('joint_cam', targets['joint_cam'].shape)

    assert 'smpl_joint_cam' in targets
    assert list(targets['smpl_joint_cam'].shape) == [33,3]
    #print ('smpl_joint_cam', targets['smpl_joint_cam'].shape)

    assert 'smpl_pose' in targets
    assert list(targets['smpl_pose'].shape) == [72,]
    #print ('smpl_pose', targets['smpl_pose'].shape)

    assert 'smpl_shape' in targets
    assert list(targets['smpl_shape'].shape) == [10,]
    #print ('smpl_shape', targets['smpl_shape'].shape)



    #print ('#####################################################')
    assert 'joint_valid' in meta_info
    assert list(meta_info['joint_valid'].shape) == [33,1]
    #print ('joint_valid', meta_info['joint_valid'].shape)

    assert 'joint_trunc' in meta_info
    assert list(meta_info['joint_trunc'].shape) == [33,1]
    #print ('joint_trunc', meta_info['joint_trunc'].shape)

    assert 'smpl_joint_trunc' in meta_info
    assert list(meta_info['smpl_joint_trunc'].shape) == [33,1]
    #print ('smpl_joint_trunc', meta_info['smpl_joint_trunc'].shape)

    assert 'smpl_joint_valid' in meta_info
    assert list(meta_info['smpl_joint_valid'].shape) == [33,1]
    #print ('smpl_joint_valid', meta_info['smpl_joint_valid'].shape)

    assert 'smpl_pose_valid' in meta_info
    assert list(meta_info['smpl_pose_valid'].shape) == [72,]
    #print ('smpl_pose_valid', meta_info['smpl_pose_valid'].shape)

    assert 'smpl_shape_valid' in meta_info
    assert type(meta_info['smpl_shape_valid']) == float
    #print ('smpl_shape_valid', meta_info['smpl_shape_valid'].shape)

    assert 'is_3D' in meta_info
    assert type(meta_info['is_3D']) == float
    #print ('is_3D', meta_info['is_3D'].shape)


def generate_segm(img2, vertices, vertex_colors, focal, princpt):

    mesh = trimesh.Trimesh(vertices, smpl.face, process=False, vertex_colors=vertex_colors)
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    renderer = pyrender.OffscreenRenderer(viewport_width=img2.shape[1], viewport_height=img2.shape[0], point_size=1.0)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    mesh = pyrender.Mesh.from_trimesh(mesh, material=None, smooth=False)

    scene.add(mesh, 'mesh')
    #focal, princpt = list(cam_param['focal'], cam_param['princpt']
    camera = pyrender.IntrinsicsCamera(fx=focal[0], fy=focal[1], cx=princpt[0], cy=princpt[1])
    scene.add(camera)


    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb_ori = rgb.copy()
    rgb = rgb[:,:,:3].astype(np.float32)
    valid_mask = (depth > 0)[:,:,None]

    # save to image
    img2 = rgb * valid_mask + img2 * (1-valid_mask)

    return rgb_ori, img2


def test_data_augm(dataset, idx, i):

    inputs, targets, meta_info = dataset[idx]
    smpl_joint_img = targets['smpl_joint_img']
    img = inputs['img']

    _tmp = smpl_joint_img.copy()
    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    _img = np.ascontiguousarray(img.numpy().transpose(1,2,0)[:,:,::-1] * 255).astype(np.uint8)
    #print (('img shape', _img.shape, _img.dtype))
    _img = vis_keypoints(_img, _tmp, radius=1, add_text=True)
    cv2.imwrite('out/smpl_keypoints_' + str(idx) + '_' + str(i) + '.jpg', _img)

def process(dataset, idx):

    inputs, targets, meta_info = dataset[idx]
    #print (inputs)
    #print (targets)
    #print (meta_info)
    print ('smpl_joint_img', targets['smpl_joint_img'].shape)
    print ('smpl_pose', targets['smpl_pose'].shape)
    print ('smpl_shape', targets['smpl_shape'].shape)

    if 'debug' in meta_info:
        print ('Camara Param', meta_info['debug']['cam_param'])

    check_gt_params(inputs, targets, meta_info)


    smpl_joint_img = targets['smpl_joint_img']
    img = inputs['img']

    _tmp = smpl_joint_img.copy()
    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    _img = np.ascontiguousarray(img.numpy().transpose(1,2,0)[:,:,::-1] * 255).astype(np.uint8)
    #print (('img shape', _img.shape, _img.dtype))
    _img = vis_keypoints(_img, _tmp, radius=1, add_text=True)
    cv2.imwrite('out/smpl_keypoints_' + str(idx) + '.jpg', _img)
    
    _img = np.ascontiguousarray(img.numpy().transpose(1,2,0)[:,:,::-1] * 255).astype(np.uint8)
    _img = vis_keypoints(_img, _tmp, kps_lines=skeleton, radius=1)
    cv2.imwrite('out/smpl_skeleton_' + str(idx) + '.jpg', _img)


    joint_img = targets['joint_img']
    joint_valid = meta_info['joint_valid']
    _tmp = joint_img.copy()
    _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
    _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
    _img = np.ascontiguousarray(img.numpy().transpose(1,2,0)[:,:,::-1] * 255).astype(np.uint8)
    #print (('img shape', _img.shape, _img.dtype))
    _img = vis_keypoints(_img, _tmp, radius=1, add_text=True, font_scale=1)
    cv2.imwrite('out/2d_keypoints_' + str(idx) + '.jpg', _img)

    _img = np.ascontiguousarray(img.numpy().transpose(1,2,0)[:,:,::-1] * 255).astype(np.uint8)
    _img = vis_keypoints(_img, _tmp, kps_lines=skeleton, radius=1)
    cv2.imwrite('out/2d_skeleton_' + str(idx) + '.jpg', _img)

    if 'debug' in meta_info:
        debug_info = meta_info['debug']

        if 'bbox' in debug_info:
            bbox = debug_info['bbox']
            _img2 = load_img(debug_info['img_path'])[:,:,::-1]

            #print ('Img 2 shape', _img2.shape, self.cfg.input_img_shape)
            _tmp[:,0] = _tmp[:,0] / cfg.input_img_shape[1] * bbox[2] + bbox[0] #* _img2.shape[1] / self.cfg.input_img_shape[1]
            _tmp[:,1] = _tmp[:,1] / cfg.input_img_shape[0] * bbox[3] + bbox[1] #* _img2.shape[0] / self.cfg.input_img_shape[0]

            #print ('GT vs Model', _tmp[0], _tmp2[0])
            _img = vis_keypoints(_img2, _tmp, radius=2, kps_valid=joint_valid, add_text=True, font_scale=1)
            cv2.imwrite(osp.join("out", 'original_2D_keypoint_' + str(idx) + '.jpg'), _img)
            
            _img2 = load_img(debug_info['img_path'])[:,:,::-1]
            _img = vis_keypoints(_img2, _tmp, radius=2, kps_valid=joint_valid, kps_lines=skeleton, add_text=False)
            cv2.imwrite(osp.join("out", 'original_2D_skeleton_' + str(idx) + '.jpg'), _img)

        if 'ori_2djoints' in debug_info:
            _img2 = load_img(debug_info['img_path'])[:,:,::-1]
            _img = vis_keypoints(_img2, debug_info['ori_2djoints'], radius=4, font_scale=1, add_text=True)
            cv2.imwrite(osp.join("out", 'original_2D_joints_' + str(idx) + '.jpg'), _img)
        if 'ori_2djoints_valid' in debug_info:
            #print (debug_info['ori_2djoints_valid'])
            pass


        cam_param = debug_info['cam_param']
        smpl_mesh_cam_orig = debug_info['smpl_mesh_cam_orig']
        img2 = load_img(debug_info['img_path'])[:,:,::-1]
        focal = list(cam_param['focal'])
        princpt = list(cam_param['princpt'])
        img2 = render_mesh(img2, smpl_mesh_cam_orig, smpl.face, {'focal': focal, 'princpt': princpt})
        #img = cv2.resize(img, (512,512))
        cv2.imwrite('out/mesh_' + str(idx) + '.jpg', img2)


        save_obj(smpl_mesh_cam_orig, smpl.face, 'out/mesh_' + str(idx) + '.obj')


        vertices = smpl_mesh_cam_orig
        part_segm = json.load(open("smpl_vert_segmentation.json"))
        vertex_colors = part_segm_to_vertex_colors(part_segm, vertices.shape[0])
        rgb, img2res = generate_segm(img2.copy(), vertices, vertex_colors, focal, princpt)

        cv2.imwrite('out/segmentation_parts_' + str(idx) + '.jpg', rgb)
        cv2.imwrite('out/segmentation_parts_img' + str(idx) + '.jpg', img2res)


        vertex_colors = binarysegm_to_vertex_colors(part_segm, vertices.shape[0])

        rgb, img2res = generate_segm(img2.copy(), vertices, vertex_colors, focal, princpt)

        cv2.imwrite('out/binsegmentation_parts_' + str(idx) + '.jpg', rgb)
        cv2.imwrite('out/binsegmentation_parts_img' + str(idx) + '.jpg', img2res)

        #mesh.show(background=(0,0,0,0))

    #mesh_out = out['smpl_mesh_cam'] * 1000 # meter to milimeter
    #rendered_img = render_mesh(inputs['img'], targets['smpl_shape'], smpl.face, {'focal': cfg.focal, 'princpt': cfg.princpt})
    #cv2.imwrite('render_cropped_img_body.jpg', rendered_img)
    #img = vis_mesh(img, mesh_out_img, 0.5)
    #cv2.imwrite(filename + '.jpg', img)


if __name__ == "__main__":
    main()