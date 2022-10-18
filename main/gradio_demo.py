"""
MEEV
Copyright (c) 2022-present NAVER Corp.
MIT License
"""

import sys
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as transforms


sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from utils.preprocessing import process_bbox, generate_patch_image
from utils.human_models import smpl
from utils.vis import render_mesh

import gradio as gr

the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = torch.jit.load('meev.pt')
model = model.to(the_device)

cfg.set_args("0", 'body')

transform = transforms.ToTensor()

def predict(inp):
  with torch.no_grad():
    original_img_height, original_img_width = inp.size
    inp = np.array(inp)

    # prepare bbox
    bbox = [0, 0, original_img_width, original_img_height] # xmin, ymin, width, height
    bbox = process_bbox(cfg, bbox, original_img_width, original_img_height)
    img, _, _ = generate_patch_image(cfg, inp, bbox, 1.0, 0.0, False, cfg.input_img_shape, 0.0, 0.0) 
    img = transform(img.astype(np.float32))/255
    img = img[None,:,:,:]
    img = img.to(the_device)

    smpl_pose, smpl_shape, smpl_mesh_cam, joint_img, smpl_joint_proj = model(img)
    mesh = smpl_mesh_cam.detach().cpu().numpy()[0]

    vis_img = img.cpu().numpy()[0].transpose(1,2,0).copy() * 255
    rendered_img = render_mesh(vis_img, mesh, smpl.face, {'focal': cfg.focal, 'princpt': cfg.princpt})
    rendered_img = rendered_img.astype(np.uint8)
  return rendered_img

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Image(type="pil"),
             examples=["sample1.jpg", "sample2.jpg", "sample3.jpg", "sample4.jpg"]).launch(share=True)

