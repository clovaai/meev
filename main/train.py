import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from config import cfg

from base import Tester, Trainer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--parts', type=str, dest='parts')
    parser.add_argument('--continue', dest='continue_train', action='store_true')
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-dv', type=int, default=None)

    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"
 
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    assert args.parts, 'Please enter human parts among [body, hand, face]'
    return args

def main():
    
    #torch.multiprocessing.set_start_method('spawn')

    # argument parse and create log
    args = parse_args()
    output_dir = args.output
    cfg.set_args(args.gpu_ids, args.parts, args.continue_train, output_dir)
    cfg.load(args.config)
    cfg.debug = args.debug
    if args.dv is not None:
        cfg.load_dataset_version(args.dv)

    cudnn.benchmark = True
    
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()

    # train
    trainer.logger.info('Start training ...')
    print ('    Batch Size : ' + str(cfg.train_batch_size))
    print ('    Epochs : ', str(trainer.start_epoch), ' - ', str(cfg.end_epoch))
    print ('    Number of GPU(s) : ', str(cfg.num_gpus), ' [' , cfg.gpu_ids, ']')

    for epoch in range(trainer.start_epoch, cfg.end_epoch):
        
        trainer.set_lr(epoch)
        trainer.tot_timer.tic()
        trainer.read_timer.tic()
        trainer.model.train()
        for itr, (inputs, targets, meta_info) in enumerate(trainer.batch_generator):
            trainer.read_timer.toc()
            trainer.gpu_timer.tic()

            # forward
            trainer.optimizer.zero_grad()

            
            loss = trainer.model(inputs, targets, meta_info, 'train', extra_info= {'epoch': epoch})
            loss = {k:loss[k].mean() for k in loss}

            sum_loss = sum(loss[k] for k in loss)
            # backward
            sum_loss.backward()
            trainer.optimizer.step()

            trainer.gpu_timer.toc()
            screen = [
                'Epoch %d/%d itr %d/%d:' % (epoch, cfg.end_epoch, itr, trainer.itr_per_epoch),
                'lr: %g' % (trainer.get_lr()),
                'speed: %.2f(%.2fs r%.2f)s/itr' % (
                    trainer.tot_timer.average_time, trainer.gpu_timer.average_time, trainer.read_timer.average_time),
                '%.2fh/epoch' % (trainer.tot_timer.average_time / 3600. * trainer.itr_per_epoch),
                ]
            screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]

            trainer.logger.info(' '.join(screen))

            trainer.tot_timer.toc()
            trainer.tot_timer.tic()
            trainer.read_timer.tic()
        

        dict_to_save = {
            'epoch': epoch,
            'network': trainer.model.state_dict(),
            'optimizer': trainer.optimizer.state_dict()}
        trainer.save_model(dict_to_save, epoch)


if __name__ == "__main__":
    main()
