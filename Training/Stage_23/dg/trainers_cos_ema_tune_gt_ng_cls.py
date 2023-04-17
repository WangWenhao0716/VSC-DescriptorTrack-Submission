from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import CrossEntropyLabelSmooth#, CosfacePairwiseLoss
from .utils.meters import AverageMeter
from .layer import MarginCosineProduct

import numpy as np

class Trainer(object):
    def __init__(self, model, model_support, num_classes, num_classes_small, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.model_support = model_support
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        self.criterion_ce_1 = CrossEntropyLabelSmooth(num_classes, epsilon=0).cuda()
        self.criterion_ce_small = CrossEntropyLabelSmooth(num_classes_small, epsilon=0).cuda()
        #self.criterion_support = nn.MSELoss().cuda()#nn.L1Loss().cuda() #nn.MSELoss().cuda()
        #self.criterion_cos_pair = CosfacePairwiseLoss(m=0.35, s=64).cuda()
        #self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        #self.w_ce = 10
        #self.w_tri = 1

        #print("The weight for loss_ce is ", self.w_ce)
        #print("The weight for loss_tri is ", self.w_tri)


    def train(self, epoch, data_loader, data_loader_small, data_loader_support, data_loader_support_T, data_loader_support_O, idx_to_class, my_dict_q, my_dict_r, optimizer, ema, train_iters=200, print_freq=1):
        self.model.eval()
        self.model_support.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_ce_1 = AverageMeter()
        losses_ce_small = AverageMeter()
        losses_ce_small_1 = AverageMeter()
        losses_sp = AverageMeter()
        losses_gt = AverageMeter()
        losses_ng = AverageMeter()
        
        precisions = AverageMeter()
        precisions_1 = AverageMeter()
        precisions_small = AverageMeter()
        precisions_small_1 = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            
            source_inputs = data_loader.next()
            source_inputs_small = data_loader_small.next()
            
            support_image, _ = data_loader_support.next()
            support_image_T, _1 = data_loader_support_T.next()
            support_image_O, _2 = data_loader_support_O.next()
            
            _1_idx = [idx_to_class[int(class_)] for class_ in _1]
            _1_name = [my_dict_q[idx] for idx in _1_idx]
            _1_name = np.array(_1_name)
            
            _2_idx = [idx_to_class[int(class_)] for class_ in _2]
            _2_name = [my_dict_r[idx] for idx in _2_idx]
            _2_name = np.array(_2_name)
            
            length = len(_1_name)
            
            _1_name_o = np.repeat(_1_name, length).reshape(length,length)
            _1_name_t = _1_name_o.T
            matrix_q = (_1_name_o == _1_name_t) * 1000000
            matrix_q = matrix_q.astype(np.float32)
            #print("matrix_q: ", matrix_q)
            
            _2_name_o = np.repeat(_2_name, length).reshape(length,length)
            _2_name_t = _2_name_o.T
            matrix_r = (_2_name_o == _2_name_t) * 1000000
            matrix_r = matrix_r.astype(np.float32)
            #print("matrix_r: ", matrix_r)
            
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            s_inputs_small, targets_small = self._parse_data(source_inputs_small)
            
            s_features, s_cls_out, s_cls_out_1, _, _ = self.model(s_inputs, targets, targets_small)
            s_features_small, _, _, s_cls_out_small, s_cls_out_small_1 = self.model(s_inputs_small, targets, targets_small)
            
            support_features  = self.model_support(support_image)
            
            ori_features, _, _, _, _ = self.model(support_image, targets, targets_small)
            ori_features_T, _, _, _, _ = self.model(support_image_T, targets, targets_small)
            ori_features_O, _, _, _, _ = self.model(support_image_O, targets, targets_small)
            
            

            # backward main #
            loss_ce, loss_ce_1, prec, prec_1 = self._forward(s_features, s_cls_out, s_cls_out_1, targets)
            loss_ce_small, loss_ce_small_1, prec_small, prec_small_1 = self._forward(s_features_small, s_cls_out_small, s_cls_out_small_1, targets_small)
            
            ori_features = ori_features/torch.norm(ori_features, dim=1).view(ori_features.shape[0],1)
            support_features = support_features/torch.norm(support_features, dim=1).view(support_features.shape[0],1)
            loss_sp = torch.mean(torch.sum((ori_features - support_features)**2, dim=1))
            
            ori_features_T = ori_features_T/torch.norm(ori_features_T, dim=1).view(ori_features_T.shape[0],1)
            ori_features_O = ori_features_O/torch.norm(ori_features_O, dim=1).view(ori_features_O.shape[0],1)
            loss_gt = torch.mean(torch.sum((ori_features_T - ori_features_O)**2, dim=1))
            
            dist_q = 2 - 2 * (ori_features_T@ori_features_T.T)
            #print("dist_q_1:", dist_q)
            dist_q = dist_q + torch.Tensor(matrix_q).cuda()
            #print("dist_q_2:", dist_q)
            loss_q = torch.mean(torch.min(dist_q, dim=1)[0])
            
            dist_r = 2 - 2 * (ori_features_O@ori_features_O.T)
            #print("dist_r_1:", dist_r)
            dist_r = dist_r + torch.Tensor(matrix_r).cuda()
            #print("dist_r_2:", dist_r)
            loss_r = torch.mean(torch.min(dist_r, dim=1)[0])
            
            loss_ng = (loss_q + loss_r)/2
            
            loss = loss_ce + loss_ce_1 + loss_ce_small + loss_ce_small_1 + 100 * loss_sp + 1 * (loss_gt - loss_ng)

            losses_ce.update(loss_ce.item())
            losses_ce_1.update(loss_ce_1.item())
            losses_ce_small.update(loss_ce_small.item())
            losses_ce_small_1.update(loss_ce_small_1.item())
            losses_sp.update(loss_sp.item())
            losses_gt.update(loss_gt.item())
            losses_ng.update(loss_ng.item())
            precisions.update(prec)
            precisions_1.update(prec_1)
            precisions_small.update(prec_small)
            precisions_small_1.update(prec_small_1)

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
                      'Loss_ce_small {:.3f} ({:.3f})\t'
                      'Loss_ce_small_1 {:.3f} ({:.3f})\t'
                      'Loss_sp {:.3f} ({:.3f})\t'
                      'Loss_gt {:.3f} ({:.3f})\t'
                      'Loss_ng {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%}) \t'
                      'Prec_1 {:.2%} ({:.2%}) \t'
                      'Prec_small {:.2%} ({:.2%}) \t'
                      'Prec_small_1 {:.2%} ({:.2%}) \t'
                      .format(epoch, i + 1, train_iters,optimizer.param_groups[0]["lr"],
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_ce_1.val, losses_ce_1.avg,
                              losses_ce_small.val, losses_ce_small.avg,
                              losses_ce_small_1.val, losses_ce_small_1.avg,
                              losses_sp.val, losses_sp.avg,
                              losses_gt.val, losses_gt.avg,
                              losses_ng.val, losses_ng.avg,
                              precisions.val, precisions.avg,
                              precisions_1.val, precisions_1.avg,
                              precisions_small.val, precisions_small.avg,
                              precisions_small_1.val, precisions_small_1.avg))

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


