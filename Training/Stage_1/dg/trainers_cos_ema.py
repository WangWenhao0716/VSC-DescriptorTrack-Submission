from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth#, CosfacePairwiseLoss
from .utils.meters import AverageMeter
from .layer import MarginCosineProduct

class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        self.criterion_ce_1 = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        #self.criterion_cos_pair = CosfacePairwiseLoss(m=0.35, s=64).cuda()
        #self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        #self.w_ce = 10
        #self.w_tri = 1

        #print("The weight for loss_ce is ", self.w_ce)
        #print("The weight for loss_tri is ", self.w_tri)


    def train(self, epoch, data_loader, optimizer, ema, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ce_1 = AverageMeter()
        #losses_cos_pair = AverageMeter()
        #losses_tr = AverageMeter()
        precisions = AverageMeter()
        precisions_1 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_features, s_cls_out, s_cls_out_1= self.model(s_inputs, targets)
            

            # backward main #
            #loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss_ce, loss_ce_1, prec, prec_1 = self._forward(s_features, s_cls_out, s_cls_out_1, targets)
            #loss = self.w_ce * loss_ce + self.w_tri * loss_tr
            loss = loss_ce + loss_ce_1

            losses_ce.update(loss_ce.item())
            losses_ce_1.update(loss_ce_1.item())
            #losses_tr.update(loss_tr.item())
            #losses_cos_pair.update(loss_cos_pair.item())
            precisions.update(prec)
            precisions_1.update(prec_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'LR:{:.8f}\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_ce_1 {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%}) \t'
                      'Prec_1 {:.2%} ({:.2%}) \t'
                      .format(epoch, i + 1, train_iters,optimizer.param_groups[0]["lr"],
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ce_1.val, losses_ce_1.avg,
                              precisions.val, precisions.avg,
                              precisions_1.val, precisions_1.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, s_outputs_1, targets):
        s_features = s_features.cuda()
        s_outputs = s_outputs.cuda()
        s_outputs_1 = s_outputs_1.cuda()
        targets = targets.cuda()
        loss_ce = self.criterion_ce(s_outputs, targets)
        loss_ce_1 = self.criterion_ce(s_outputs_1, targets)
        
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]
        
        prec_1, = accuracy(s_outputs_1.data, targets.data)
        prec_1 = prec_1[0]

        return loss_ce, loss_ce_1, prec, prec_1


