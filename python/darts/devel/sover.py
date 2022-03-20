'''
File: sover.py
Project: devel
File Created: Saturday, 8th January 2022 9:10:15 pm
Author: Xu En (xuen@mokar.com)
-----
Last Modified: Saturday, 8th January 2022 9:11:01 pm
Modified By: Xu En (xuen@mokahr.com)
-----
Copyright 2021 - 2022 Your Company, Moka
'''

import os
import time
import numpy as np
import torch
from torch import optim
import h5py

if not torch.cuda.is_available():
    print("WARN: NO CUDA DEVICE!")


def saveTorchModel(model, out_dir, epoch_num):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # save model for torch
    torch.save(model, os.path.join(out_dir, "model-%d.pkl" % epoch_num))
    # save model for numpy
    state_dict = dict((k, v.cpu().numpy()) for (k, v) in model.state_dict().items())
    np.savez(os.path.join(out_dir, "model-%d" % epoch_num), **state_dict)
    # save model for h5
    with h5py.File(os.path.join(out_dir, "model-%d.h5" % epoch_num), 'w') as hf:
        for k, v in state_dict.items():
            hf.create_dataset(k,  data=v)
    print("Write model to {}".format(out_dir))


def loadInitModel(model, modeldir, filename="model-init.npz"):
    model_path = os.path.join(modeldir, filename)
    if not os.path.exists(model_path):
        return
    param_dict = dict((k, torch.from_numpy(v)) for k, v in np.load(model_path).items())
    print("Load model from {}".format(model_path))
    for name, p in model.named_parameters():
        if name not in param_dict:
            print("param [%s] not in dict,we skip it" % name)
            continue
        give = param_dict[name]
        if give.shape != p.data.shape:
            print("param [%s] init value shape is %s not equal in dict,we skip it" % (name, str(p.data.shape)))
            continue
        p.data = give.to(p.data.device)


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
        loadInitModel(self.model, modeldir, filename=conf.get("initfile", "model-init.npz"))
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
        _step, _epoch, _sample = 0, 0, 0
        self.optimizer.zero_grad()

        def apply_grad():
            # apply loss and clear grad
            self.optimizer.step()
            self.optimizer.zero_grad()

        # save test before start
        saveTorchModel(self.model, self.out_dir, _epoch)

        # iter data
        while _epoch < self.epoch_num:
            _sample = 0
            _epoch += 1

            # iter samples
            for td in self.train_iter:
                _sample += 1
                _step += 1
                loss = self.model(*td)
                loss.backward()
                # apply loss
                apply_grad()
                print('train [epoch|step|sample]:[%d|%d|%d] loss:%g' % (_epoch, _step, _sample, float(loss.cpu())))
                if _step % 200000 == 0:
                    saveTorchModel(self.model, self.out_dir, _epoch)
            saveTorchModel(self.model, self.out_dir, _epoch)
