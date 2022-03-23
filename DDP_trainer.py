import math
import logging

# from tqdm import tqdm
import numpy as np

import os
import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
import torch.nn as nn
import torchvision.utils as vutils

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tensorboardX import SummaryWriter

import utils.util as utils
from metrics.calculate_psnr_ssim import calculate_psnr, calculate_ssim


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_epoch = 10
    final_setp = 80
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, gpu, global_rank):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.device=gpu

        self.writer = SummaryWriter(self.config.ckpt_path)

        self.model=model.cuda(gpu)
        self.global_rank=global_rank
        self.train_sampler=DistributedSampler(train_dataset, num_replicas=config.world_size, rank=global_rank)
        self.test_sampler=DistributedSampler(test_dataset, num_replicas=config.world_size, rank=global_rank)

        self.loss_func = nn.L1Loss()
        self.global_step = 0

    def save_checkpoint(self, epoch, optim, validation,save_name):
        if self.global_rank==0: ## Only save in global rank 0
            # DataParallel wrappers keep raw model object in .module attribute
            raw_model = self.model.module if hasattr(self.model, "module") else self.model
            save_url=os.path.join(self.config.ckpt_path,save_name+'.pth')
            print("saving %s"%(save_url))
            torch.save({'model': raw_model.state_dict(),
                        'epoch': epoch,
                        'optimizer':optim.state_dict(),
                        'best_val': validation}, save_url)

    def load_checkpoint(self, resume_path):
        if os.path.exists(resume_path):
            data = torch.load(resume_path, map_location='cuda:{}'.format(self.device))
            self.model.load_state_dict(data['model'])
            print('Finished reloading the Epoch %d model'%(data['epoch']))
            return data
        else:
            if self.global_rank==0:
                print('Warnning: There is no trained model found. An initialized model will be used.')
        return None

    def train(self, loaded_ckpt):

        self.previous_epoch=-1
        
        #model = DDP(self.model,device_ids=[self.global_rank],output_device=self.global_rank,broadcast_buffers=True)
        self.model = DDP(self.model,device_ids=[self.device])

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate,betas =self.config.betas,weight_decay=self.config.weight_decay)

        if loaded_ckpt is not None:
            self.optimizer.load_state_dict(loaded_ckpt['optimizer'])
            self.best_val=loaded_ckpt['best_val']
            self.previous_epoch=loaded_ckpt['epoch']
            print('Finished reloading the Epoch %d optimizer'%(loaded_ckpt['epoch']))
        else:
            print('Warnning: There is no previous optimizer found. An initialized optimizer will be used.')

                    
        ## TODO: Use different seeds to initialize each worker. (This issue is caused by the bug of pytorch itself)
        self.train_loader = DataLoader(self.train_dataset, shuffle=False, pin_memory=True,
                            batch_size=self.config.batch_size // self.config.world_size, ## BS of each GPU
                            num_workers=self.config.num_workers,sampler=self.train_sampler)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, pin_memory=True,
                            batch_size=self.config.batch_size // self.config.world_size, ## BS of each GPU
                            num_workers=self.config.num_workers,sampler=self.test_sampler)

        if loaded_ckpt is None:
            self.best_val = float('inf')
        self.cosLR = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max = 100, eta_min = 1e-6) 
        for epoch in range(self.config.max_epochs):

            if self.previous_epoch!=-1 and epoch<=self.previous_epoch:
                continue

            if epoch==self.previous_epoch+1 and self.global_rank == 0:
                print("Resume from Epoch %d"%(epoch))

            self.train_sampler.set_epoch(epoch) ## Shuffle each epoch

            epoch_start=time.time()
            self.run_epoch(epoch)
            if self.test_dataset is not None:
                val_ret = self.run_test(epoch)
            if self.global_rank==0:
                print("Epoch: %d, test ret: %f, time for one epoch: %d seconds"%(epoch, val_ret, time.time() - epoch_start))
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or val_ret < self.best_val
            if self.config.ckpt_path is not None and good_model and self.global_rank==0: ## Validation on the global_rank==0 process
                self.best_val = val_ret
                print("current best epoch is %d"%(epoch))
                self.save_checkpoint(epoch, self.optimizer, self.best_val, save_name='best')
            
            if self.global_rank == 0:
                if not np.isnan(val_ret):
                    self.save_checkpoint(epoch, self.optimizer, self.best_val,save_name='latest')
                else:
                    print('NaN happens, try to reload the previous normal checkpoint')

    def run_epoch(self, epoch):
        self.model.train()
        loss_avg = utils.AverageMeter('loss')
        scaler = GradScaler()
        for it, (img, inpaint_mask, seg_mask, img_gt) in enumerate(self.train_loader):

            # place data on the correct device
            input_img = torch.cat((img, inpaint_mask, seg_mask), dim = 1)
            input_img = input_img.to(self.device)
            img_gt = img_gt.to(self.device)

            # forward the model
            if self.config.AMP: ## use AMP
                with autocast():
                    out_img = self.model(input_img)
                    loss = self.loss_func(out_img, img_gt)
                self.model.zero_grad()
                scaler.scale(loss).backward()
                ## AMP+Gradient Clip
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)

                scaler.step(self.optimizer)
                scaler.update()
            else:
                out_img = self.model(input_img)
                loss = self.loss_func(out_img, img_gt)
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()
            loss_avg.update(loss.item())


            if self.global_step % self.config.print_freq == 0:
                loss = utils.dist_mean_param(loss_avg.avg, self.global_rank)
                if self.global_rank == 0:
                    print(f"epoch {epoch+1} iter {it}: train loss {loss:.5f}.")
                    self.writer.add_scalar('train/loss', loss, self.global_step)
            if self.global_step % self.config.write_freq == 0 and self.global_rank == 0:
                img = vutils.make_grid(img, normalize=True, scale_each=True)
                self.writer.add_image('trian/input_img', img, self.global_step)
                inpaint_mask = vutils.make_grid(inpaint_mask, normalize=True, scale_each=True)
                self.writer.add_image('trian/inpaint_mask', inpaint_mask, self.global_step)
                seg_mask = vutils.make_grid(seg_mask, normalize=True, scale_each=True)
                self.writer.add_image('trian/seg_mask', seg_mask, self.global_step)
                out_img = vutils.make_grid(out_img, normalize=True, scale_each=True)
                self.writer.add_image('trian/out_img', out_img, self.global_step)
                gt = vutils.make_grid(img_gt, normalize=True, scale_each=True)
                self.writer.add_image('trian/gt', gt, self.global_step)
     
            self.cosLR.step()
            self.global_step += 1
    
    def run_test(self, epoch):
        self.model.eval()
        psnr_avg = utils.AverageMeter('psnr')
        ssim_avg = utils.AverageMeter('ssim')
        loss_avg = utils.AverageMeter('loss')
        for it, (img, inpaint_mask, seg_mask, img_gt) in enumerate(self.test_loader):
            # place data on the correct device
            input_img = torch.cat((img, inpaint_mask, seg_mask), dim = 1)
            input_img = input_img.to(self.device)
            img_gt = img_gt.to(self.device)

            with torch.no_grad():
                out_img = self.model(input_img)
            loss_avg.update(self.loss_func(out_img, img_gt).item())
            psnr_avg.update(calculate_psnr(out_img, img_gt))
            ssim_avg.update(calculate_ssim(out_img, img_gt))
        loss = utils.dist_mean_param(loss_avg.avg, self.global_rank)
        psnr = utils.dist_mean_param(psnr_avg.avg, self.global_rank)
        ssim = utils.dist_mean_param(ssim_avg.avg, self.global_rank)
        if self.global_rank == 0:
            print("test loss: {:.5f}, psnr: {:.5f}, ssim: {:.5f}".format(loss, psnr, ssim))
            self.writer.add_scalar('val/loss', loss, epoch)
            self.writer.add_scalar('val/psnr', psnr, epoch)
            self.writer.add_scalar('val/ssim', ssim, epoch)
            img = vutils.make_grid(img, normalize=True, scale_each=True)
            self.writer.add_image('val/input_img', img, epoch)
            inpaint_mask = vutils.make_grid(inpaint_mask, normalize=True, scale_each=True)
            self.writer.add_image('val/inpaint_mask', inpaint_mask, epoch)
            seg_mask = vutils.make_grid(seg_mask, normalize=True, scale_each=True)
            self.writer.add_image('val/seg_mask', seg_mask, epoch)
            out_img = vutils.make_grid(out_img, normalize=True, scale_each=True)
            self.writer.add_image('val/out_img', out_img, epoch)
            gt = vutils.make_grid(img_gt, normalize=True, scale_each=True)
            self.writer.add_image('val/gt', gt, epoch)
        return psnr + ssim
