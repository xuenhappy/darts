# coding=utf-8
'''
File: torchsover.py
Project: workAbout
File Created: Thursday, 18th June 2020 4:57:40 pm
Author: enxu (xuen@mokar.com)
-----
Last Modified: Thursday, 18th June 2020 4:57:44 pm
Modified By: enxu (xuen@mokahr.com)
-----
Copyright 2021 - 2020 Your Company, Moka
'''
import sys
import os
import time
import math
import numpy as np
import torch
from torch import optim
from .torchtools import drop_module_grad


def save_torch_model(model, out_dir, epoch_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # save model for torch
    torch.save(model, os.path.join(out_dir, "model-%d.pkl" % epoch_num))
    # save model for numpy
    state_dict = dict((k, v.cpu().numpy()) for (k, v) in model.state_dict().items())
    np.savez(os.path.join(out_dir, "model-%d" % epoch_num), **state_dict)
    print("Write model to {}".format(out_dir))


def load_init_model(model, modeldir, filename="model-init.npz"):
    model_path = os.path.join(modeldir, filename)
    if not os.path.exists(model_path):
        return
    param_dict = dict((k, torch.from_numpy(v)) for k, v in np.load(model_path).items())
    model.load_state_dict(param_dict)
    print("Load model from {}".format(model_path))


class TeachSolver():
    def __init__(self, model, train_iter, conf={}):
        self.train_iter = train_iter
        for (k, v) in conf.items():
            setattr(self, k, v)

        self.model = model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        modeldir = os.path.abspath(os.path.join(os.path.curdir, self.model_outdir))
        self.out_dir = os.path.join(modeldir, str(int(time.time())))
        load_init_model(self.model, modeldir)

        self.max_grad_csum_step = 10 if not hasattr(self, 'max_grad_csum_step') else self.max_grad_csum_step
        self.max_grad_csum_step = max(1, self.max_grad_csum_step)

        self.lrate = 0.001 if not hasattr(self, "lrate") else self.lrate
        self.lrate = max(1e-4, self.lrate)

        weight_p, bias_p = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in name) or ('bnorm' in name):
                bias_p.append(p)
                continue
            weight_p.append(p)

        self.optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': 1e-5},
            {'params': bias_p, 'weight_decay': 0}
        ], lr=self.lrate)

    def solve(self):
        # init flag data
        step, epoch_num, sample_num = 0, 0, 0
        losses = []
        self.optimizer.zero_grad()

        def apply_grad():
            alpha = 1/math.sqrt(len(losses))
            for p in self.model.parameters():
                if not p.requires_grad:
                    continue
                if hasattr(p, 'drop_grad') and p.drop_grad:
                    p.grad = None
                if p.grad is None:
                    continue
                p.grad *= alpha
            # apply loss and clear grad
            self.optimizer.step()
            self.optimizer.zero_grad()
            avg_loss = sum(losses)/len(losses)
            losses.clear()
            return avg_loss

        # iter data
        while epoch_num < self.epoch_num:
            sample_num = 0
            epoch_num += 1
            # test if could write and sample could use
            self.train_iter.reset()
            save_torch_model(self.model, self.out_dir, epoch_num)
            # iter samples
            print("start %d ecoph iter train" % epoch_num)
            for td in self.train_iter:
                sample_num += 1
                step += 1
                # csum loss
                loss = self.model(*td)
                loss.backward()
                losses.append(float(loss.cpu()))
                # apply loss
                if len(losses) >= self.max_grad_csum_step:
                    avg_loss = apply_grad()
                    print('train [epoch|step|sample]:[%d|%d|%d] loss:%g' % (epoch_num, step, sample_num, avg_loss))

                if step % 200000 == 0:
                    save_torch_model(self.model, self.out_dir, epoch_num)

        # last update
        if len(losses) > 0:
            apply_grad()
        save_torch_model(self.model, self.out_dir, epoch_num)
