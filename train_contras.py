import numpy as np
import torch
import torch.nn as nn
import time

from torch.nn.utils import clip_grad_norm

class ContrastiveLoss(nn.Module):
    """ Contrastive Loss """
    def __init__(self, m1=1., m2=1.):
        super(ContrastiveLoss, self).__init__()
        self.m1 = m1
        self.m2 = m2
        self.cr_en = nn.CrossEntropyLoss(reduce=False)
    def forward(self, n_pred, r_pred, s_pred, target):
        cr_en_n = self.cr_en(n_pred, target)
        cr_en_r = self.cr_en(r_pred, target)
        cr_en_s = self.cr_en(s_pred, target)
        # loss_r = torch.max(0., self.m1 - cr_en_r)
        # loss_s = torch.max(0., self.m2 - cr_en_s)
        loss_r = torch.clamp(self.m1 - cr_en_r, min=0.)
        loss_s = torch.clamp(self.m2 - cr_en_s, min=0.)
        # loss = cr_en_n + loss_r + loss_s
        loss = torch.mean(cr_en_n + loss_r + loss_s)
        return loss


def train_contrastive(train_loader, model, 
        criterion, optimizer, epoch, train_logger, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (n_inputs, r_inputs, s_inputs, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        n_inputs_var = torch.autograd.Variable(n_inputs)
        r_inputs_var = torch.autograd.Variable(r_inputs)
        s_inputs_var = torch.autograd.Variable(s_inputs)
        target_var = torch.autograd.Variable(target)

        # compute output
        n_outputs = model(n_inputs_var)
        r_outputs = model(r_inputs_var)
        s_outputs = model(s_inputs_var)
        loss = criterion(n_outputs, r_outputs, s_outputs, target_var)
        # print('n_outputs:', n_outputs)
        # print('r_outputs:', r_outputs)
        # print('s_outputs:', s_outputs)
        # print('loss:', loss)
        # input('...')
        # loss = criterion(n_outputs, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(n_outputs.data, target, topk=(1,5))
        losses.update(loss.data[0], n_inputs.size(0))
        top1.update(prec1[0], n_inputs.size(0))
        top5.update(prec5[0], n_inputs.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, 
                    args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_line = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, 
                   top5=top5, lr=optimizer.param_groups[-1]['lr']))
            print(log_line)
            # train_logger.write(log_line+'\n')
            with open(train_logger, 'a') as f:
                f.write(log_line+'\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
