from collections import OrderedDict
import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader, Subset
import torch.optim
import torchvision.transforms as transforms
from timer import Timer
from logger import colorlogger
from torch.nn.parallel.data_parallel import DataParallel
from model import freeze_model, get_model
from dataset import MultipleDatasets
from utils.cfg_utils import getAnyFromCfg, getIntFromCfg, getStringFromCfg
from utils.dir import add_pypath
from utils.format_utils import float_to_2decimal
import random
import numpy as np
from logger import mainlogger


the_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# seed
def set_seed(SEED=0):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # cudnn related setting
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = False

class Base(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, cfg, log_name='logs.txt'):
        self.cfg = cfg

        self.cur_epoch = 0

        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()

        # logger
        self.logger = colorlogger(cfg.log_dir, log_name=log_name)

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return

class Trainer(Base):
    def __init__(self, cfg):
        super(Trainer, self).__init__(cfg, log_name = 'train_logs.txt')
        set_seed(74)

    def get_optimizer(self, model):
        total_params = []
        for module in model.module.trainable_modules:
            total_params += list(module.parameters())
        optimizer = torch.optim.Adam(total_params, lr=self.cfg.lr)
        return optimizer

    def save_model(self, state, epoch):
        file_path = osp.join(self.cfg.model_dir,'snapshot_{}.pth.tar'.format(str(epoch)))

        # do not save human model layer weights
        dump_key = []
        for k in state['network'].keys():
            if 'smpl_layer' in k or 'mano_layer' in k or 'flame_layer' in k:
                dump_key.append(k)
        for k in dump_key:
            state['network'].pop(k, None)

        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, cur_epoch = None):
        model_file_list = glob.glob(osp.join(self.cfg.model_dir,'*.pth.tar'))
        if cur_epoch is None:
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
        ckpt_path = osp.join(self.cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
        ckpt = torch.load(ckpt_path, map_location=the_device) 
        start_epoch = ckpt['epoch'] + 1
        sd = ckpt['network']
        # filter weights in case of EgoBodyVideo ( size of decoder missmatch )
        if self.cfg.k > 1:
            new_state_dict = OrderedDict()
            for k, v in sd.items():
                if str(k).startswith('module.position_net.conv.0') or str(k).startswith('module.rotation_net.shape_out.0') \
                    or str(k).startswith('module.rotation_net.cam_out.0') or str(k).startswith('module.rotation_net.conv.0'):
                    pass
                else:
                    new_state_dict[k] = v
            sd = new_state_dict

        model.load_state_dict(sd, strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])    

        self.logger.info('Load checkpoint from {}'.format(ckpt_path))
        return start_epoch, model, optimizer

    def set_lr(self, epoch):
        for e in self.cfg.lr_dec_epoch:
            if epoch < e:
                break
        if epoch < self.cfg.lr_dec_epoch[-1]:
            idx = self.cfg.lr_dec_epoch.index(e)
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** idx)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = self.cfg.lr / (self.cfg.lr_dec_factor ** len(self.cfg.lr_dec_epoch))

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr
        
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating train dataset...")

        # dynamic dataset import
        for i in range(len(self.cfg.trainset_3d)):
            add_pypath(osp.join(self.cfg.data_dir, self.cfg.trainset_3d[i]))
            exec('from ' + self.cfg.trainset_3d[i] + ' import ' + self.cfg.trainset_3d[i])
        for i in range(len(self.cfg.trainset_2d)):
            add_pypath(osp.join(self.cfg.data_dir, self.cfg.trainset_2d[i]))
            exec('from ' + self.cfg.trainset_2d[i] + ' import ' + self.cfg.trainset_2d[i])


        self.trainset3d_datasets = []
        for i in range(len(self.cfg.trainset_3d)):
            start = time.time()
            self.trainset3d_datasets.append(eval(self.cfg.trainset_3d[i])(self.cfg, transforms.ToTensor(), "train"))
            end = time.time()
            self.logger.info ('Train 3D Dataset ' + str(self.cfg.trainset_3d[i]) + ' generated in ' + str(float_to_2decimal(end-start)) + ' sec.')
        self.trainset2d_datasets = []
        for i in range(len(self.cfg.trainset_2d)):
            start = time.time()
            self.trainset2d_datasets.append(eval(self.cfg.trainset_2d[i])(self.cfg, transforms.ToTensor(), "train"))
            end = time.time()
            self.logger.info ('Train 2D Dataset ' + str(self.cfg.trainset_2d[i]) + ' generated in ' + str(float_to_2decimal(end-start)) + ' sec.')
       
        valid_loader_num = 0
        if len(self.trainset3d_datasets) > 0:
            trainset3d_loader = [MultipleDatasets(self.trainset3d_datasets, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset3d_loader = []
        if len(self.trainset2d_datasets) > 0:
            trainset2d_loader = [MultipleDatasets(self.trainset2d_datasets, make_same_len=False)]
            valid_loader_num += 1
        else:
            trainset2d_loader = []

        if valid_loader_num > 1:
            if self.cfg.balance_dataset:  # default value
                trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=True)
            else:
                trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)
        else:
            trainset_loader = MultipleDatasets(trainset3d_loader + trainset2d_loader, make_same_len=False)

        self.logger.info ("All dataset loaded : " + str(trainset_loader.__len__()))

        db_max_length= getIntFromCfg(self.cfg, 'dataset_max_length', -1)
        if db_max_length > 1:
            indices = torch.arange(db_max_length)
            trainset_loader = Subset(trainset_loader, indices)
            mainlogger.warning ("Using a subset of the training dataset " + str(trainset_loader.__len__()))

        self.itr_per_epoch = math.ceil(len(trainset_loader) / self.cfg.num_gpus / self.cfg.train_batch_size)
        self.batch_generator = DataLoader(dataset=trainset_loader, batch_size=self.cfg.num_gpus*self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_thread, pin_memory=True, drop_last=True)

    def _make_model(self, cur_epoch=None):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(self.cfg, 'train')
        self.inmodel = model
       
        model = DataParallel(model)
        model = model.to(the_device)

        optimizer = self.get_optimizer(model)


        if self.cfg.continue_train:
            start_epoch, model, optimizer = self.load_model(model, optimizer, cur_epoch=cur_epoch)
        else:
            start_epoch = 0
        
        freeze_model(self.cfg, model)


        model.train()

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer



class Tester(Base):
    def __init__(self, cfg, test_epoch):
        self.test_epoch = int(test_epoch)
        super(Tester, self).__init__(cfg, log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating test dataset...")
        # dynamic dataset import

        self.testsets = []
        self.batch_generators = []
        self.dataset_test_names = self.cfg.testset
        if isinstance(self.cfg.testset, str):
            self.dataset_test_names = [self.cfg.testset]

        for i in range(len(self.dataset_test_names)):
            self.logger.info("loading test dataset " + self.dataset_test_names[i])
            add_pypath(osp.join(self.cfg.data_dir, self.dataset_test_names[i]))

            exec('from ' + self.dataset_test_names[i] + ' import ' + self.dataset_test_names[i])

            start = time.time()
            testset_loader = eval(self.dataset_test_names[i])(self.cfg, transforms.ToTensor(), "test")
            end = time.time()
            self.logger.info ('Test Dataset ' + str(self.dataset_test_names[i]) + ' generated in ' + str(float_to_2decimal(end-start)) + ' sec.')

            self.testsets.append(testset_loader)
            
            db_max_length= getIntFromCfg(self.cfg, 'dataset_max_length', -1)
            if db_max_length > 1:
                indices = torch.arange(db_max_length)
                testset_loader = Subset(testset_loader, indices)
                mainlogger.warning ("Using a subset of the test dataset " + str(testset_loader.__len__()))

            batch_generator = DataLoader(dataset=testset_loader, batch_size=self.cfg.num_gpus*self.cfg.test_batch_size, shuffle=False, num_workers=self.cfg.num_thread, pin_memory=True)
            self.batch_generators.append(batch_generator)
        

    def _make_model(self):
        model_path = os.path.join(self.cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(self.cfg, 'test')
        model = DataParallel(model)

        model = model.to(the_device)

        self.logger.info('Load checkpoint from {}'.format(model_path))
        ckpt = torch.load(model_path, map_location=the_device)
        model.load_state_dict(ckpt['network'], strict=False)
        model.eval()

        self.model = model
    
    def _evaluate(self, testset, outs, cur_sample_idx, viz=False):
        eval_result = testset.evaluate(outs, cur_sample_idx, viz=viz)
        return eval_result

    def _print_eval_result(self, testset, name: str, eval_result):
        s = testset.print_eval_result(eval_result)
        self.logger.info("Validation Results for " + name)
        self.logger.info(s)

