import numpy as np
import torchvision
import torch
import matplotlib.pyplot as plt
import logging
from utils.util import set_seed,Logger
from datas.ffhq_dataset import FfhqDatasetMask
from models.archs.restormer_arch import Restormer
from DDP_trainer import TrainerConfig,Trainer
import argparse
import os
import sys

import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp
import torch.multiprocessing as mp

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def main_worker(gpu,opts):

    rank=opts.node_rank*opts.gpus+gpu ## Get the global Rank

    torch.cuda.set_device(gpu)
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=opts.world_size,                              
    	rank=rank,
        group_name='mtorch'                                               
    )
    set_seed(42)

    if rank == 0:
        sys.stdout = Logger(os.path.join(opts.ckpt_path, 'log.txt'))


    ##TODO: directly use the provided color palette provided by OpenAI. [√]


    ## Define the dataset
    train_dataset=FfhqDatasetMask(opts.data_path,image_size=opts.image_size)
    test_dataset=FfhqDatasetMask(opts.validation_path,image_size=opts.image_size)

    model = Restormer()

    train_epochs=opts.train_epoch

    ## By default: 8xV100 GPUs
    ## TODO: Modify the ckpt path [√]
    train_config=TrainerConfig(max_epochs=train_epochs,batch_size=opts.batch_size,
                                learning_rate=opts.lr,betas = (0.9, 0.95), 
                                weight_decay=1e-4,lr_decay=True,warmup_epoch=5, 
                                final_step=80,ckpt_path=opts.ckpt_path,
                                num_workers=8,GPU_ids=opts.GPU_ids, world_size=opts.world_size,
                                AMP=opts.AMP,print_freq=opts.print_freq, write_freq = opts.write_freq)
    trainer = Trainer(model, train_dataset, test_dataset, train_config, gpu, rank)
    loaded_ckpt=trainer.load_checkpoint(opts.resume_ckpt)
    trainer.train(loaded_ckpt)
    print("Finish the training ...")



if __name__=='__main__':


    parser=argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='Inpainting',help='The name of this exp')
    parser.add_argument('--GPU_ids',type=str,default='0')
    parser.add_argument('--ckpt_path',type=str,default='./ckpt')
    parser.add_argument('--data_path',type=str,default='/home/ziyuwan/workspace/data/',help='Indicate where is the training set')
    parser.add_argument('--batch_size',type=int,default=2*6,help='16*8 maybe suitable for V100')
    parser.add_argument('--train_epoch',type=int,default=80,help='how many epochs')
    parser.add_argument('--print_freq',type=int,default=100,help='While training, the freq of printing log')
    parser.add_argument('--write_freq',type=int,default=1000,help='While training, the freq of printing log')

    parser.add_argument('--validation_path',type=str,default='',help='where is the validation set of ImageNet')

    parser.add_argument('--image_size',type=int,default=512,help='input sequence length = image_size*image_size')

    ### Define the size of transformer
    parser.add_argument('--lr',type=float,default=3e-4)


    ### DDP+AMP
    parser.add_argument('--DDP',action='store_true',help='using DDP rather than normal data parallel')
    parser.add_argument('--nodes',type=int,default=1,help='how many machines')
    parser.add_argument('--gpus',type=int,default=1,help='how many GPUs in one node')
    parser.add_argument('--node_rank',type=int,default=0,help='the id of this machine')
    parser.add_argument('--AMP',action='store_true',help='Automatic Mixed Precision')
    parser.add_argument('--resume_ckpt',type=str,default='latest.pth',help='start from where, the default is latest')
    

    opts=parser.parse_args()
    opts.ckpt_path=os.path.join(opts.ckpt_path,opts.name)
    opts.resume_ckpt=os.path.join(opts.ckpt_path,opts.resume_ckpt)
    os.makedirs(opts.ckpt_path, exist_ok=True)

    opts.world_size=opts.nodes*opts.gpus
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '48364'   


    logging.basicConfig(
            # filename=os.path.join(opts.ckpt_path,'running.log'),
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )


    mp.spawn(main_worker, nprocs=opts.gpus, args=(opts,))
