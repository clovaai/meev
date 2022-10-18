import os
import os.path as osp
import sys
import numpy as np
import json
import socket
import torch

class Config:

    def __init__(self):

        ## dataset
        self.trainset_3d = ['Human36M'] 
        self.trainset_3d = ['Human36M', 'AGORA', 'MPI_INF_3DHP'] 
        #trainset_3d = ['Human36M', 'MPI_INF_3DHP'] 
        self.trainset_2d = ['MSCOCO', 'MPII']
        self.testset = ['PW3D', 'MSCOCO', 'Human36M', 'AGORA', 'EgoBody']

        self.balance_dataset = True

        ## model setting
        self.backbone = "resnet"
        self.use_pretrained_weight = True

        ## resnet backbone settings
        self.resnet_type = 50

        ## training config
        self.lr = 1e-4
        self.lr_dec_factor = 10
        self.train_batch_size = 48

        ## testing config
        self.test_batch_size = 48

        ## others
        self.num_thread = 40
        
        self.gpu_ids = '0'
        self.num_gpus = 1
        self.parts = 'body'
        self.continue_train = False

        self.debug = False
        
        ## directory
        self.cur_dir = osp.dirname(os.path.abspath(__file__))
        self.root_dir = osp.join(self.cur_dir, '..')
        self.data_dir = osp.join(self.root_dir, 'data')
        self.hub_dir = osp.join(self.root_dir, 'hub')
        self.output_dir = osp.join(self.root_dir, 'output')
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.result_dir = osp.join(self.output_dir, 'result')    
        self.human_model_path = osp.join(self.root_dir, 'common', 'utils', 'human_model_files')
        #mainlogger.set_output_folder(self.log_dir)

    def load_dataset_version(self, version):
        self.trainset_2d = MAPPING_DATASETS[version]['2d']
        self.trainset_3d = MAPPING_DATASETS[version]['3d']
        for i in range(len(cfg.trainset_3d)):
            add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
        for i in range(len(cfg.trainset_2d)):
            add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))

    def load(self, config_file):
        if config_file is not None:
            full_config_file = osp.join(self.root_dir, config_file)
            if os.path.exists(full_config_file):
                print ('Loading configuration file', config_file)
                import yaml
                with open(full_config_file) as f:
                    conf = yaml.load(f, Loader=yaml.FullLoader)
                    if 'DATASET' in conf:
                        dconf = conf['DATASET']
                        if 'VERSION' in dconf:
                            self.load_dataset_version(dconf['VERSION'])
                        for c in dconf:
                            self.__setattr__(str(c).lower(), dconf[c])

                    if 'MODEL' in conf and conf['MODEL'] is not None:
                        mconf = conf['MODEL']
                        for c in mconf:
                            self.__setattr__(str(c).lower(), mconf[c])

                    if 'TRAINING' in conf and conf['TRAINING'] is not None:
                        tconf = conf['TRAINING']
                        for c in tconf:
                            self.__setattr__(str(c).lower(), tconf[c])
                        

                self.dump()
            else:
                print ("Configuration file not found", full_config_file)

    def set_output_dir(self, output_dir):
        self.output_dir = osp.join(self.root_dir, output_dir)
        self.model_dir = osp.join(self.output_dir, 'model_dump')
        self.vis_dir = osp.join(self.output_dir, 'vis')
        self.log_dir = osp.join(self.output_dir, 'log')
        self.result_dir = osp.join(self.output_dir, 'result')
        mainlogger.set_output_folder(self.log_dir)
    
    def set_args(self, gpu_ids, parts, continue_train=False, continue_dir="output"):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.parts = parts
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        self.set_output_dir(continue_dir)
        make_folder(cfg.model_dir)
        make_folder(cfg.vis_dir)
        make_folder(cfg.log_dir)
        make_folder(cfg.result_dir)
        if continue_train:
            print('Continue training from ', continue_dir)

        print('>>> Using GPU: {}'.format(self.gpu_ids))
        if self.parts == 'body':
            self.bbox_3d_size = 2
            self.camera_3d_size = 2.5
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
            self.lr_dec_epoch = [4, 6]
            self.end_epoch = 7
        elif self.parts == 'hand':
            self.bbox_3d_size = 0.3
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 256)
            self.output_hm_shape = (8, 8, 8)
            self.lr_dec_epoch = [10, 12] 
            self.end_epoch = 13 
        elif self.parts == 'face':
            self.bbox_3d_size = 0.3
            self.camera_3d_size = 0.4
            self.input_img_shape = (256, 192)
            self.output_hm_shape = (8, 8, 6)
            self.lr_dec_epoch = [10, 12] 
            self.end_epoch = 13 
        else:
            assert 0, 'Unknown parts: ' + self.parts
        
        self.focal = (5000, 5000) # virtual focal lengths
        self.princpt = (self.input_img_shape[1]/2, self.input_img_shape[0]/2) # virtual principal point position
        self.dump()

    def dump(self):
        # dump the config file into output
        import yaml
        with open(osp.join(self.output_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.__dict__, file)
        json_str = json.dumps(cfg, default=lambda o: o.__dict__, sort_keys=True, indent=4)
        with open(osp.join(self.output_dir, 'config.json'), 'w') as outfile:
            outfile.write(json_str)

cfg = Config()


MAPPING_DATASETS = {
    0 : { '2d': [], '3d': ['MPI_INF_3DHP'] },    # Default model setup for testing/debugging code
    1 : { '2d': ['MSCOCO', 'MPII'], '3d': ['Human36M', 'MPI_INF_3DHP'] },
    2 : { '2d': ['MSCOCO', 'MPII'], '3d': ['Human36M', 'AGORA', 'MPI_INF_3DHP']  },
}

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))

from utils.dir import add_pypath, make_folder
from logger import mainlogger

add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
for i in range(len(cfg.testset)):
    add_pypath(osp.join(cfg.data_dir, cfg.testset[i]))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)
make_folder(cfg.hub_dir)
torch.hub.set_dir(cfg.hub_dir)

