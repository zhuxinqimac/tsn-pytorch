import argparse
import os
import time
import json
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from temporal_transforms import ReverseFrames, ShuffleFrames
from opts import parser
from train_contras import train_contrastive, ContrastiveLoss

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'something':
        num_class = 174
    else:
        raise ValueError('Unknown dataset '+args.dataset)

    if args.dataset == 'something':
        img_prefix = ''
    else:
        img_prefix = 'image_'
    with open(os.path.join(args.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(args), opt_file)
    if not (args.consensus_type == 'lstm' or args.consensus_type == 'conv_lstm'):
        args.lstm_out_type = None
    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, 
                dropout=args.dropout, 
                partial_bn=not args.no_partialbn, 
                lstm_out_type=args.lstm_out_type, 
                lstm_layers=args.lstm_layers, 
                lstm_hidden_dims=args.lstm_hidden_dims, 
                conv_lstm_kernel=args.conv_lstm_kernel, 
                bi_add_clf=args.bi_add_clf, 
                bi_out_dims=args.bi_out_dims, 
                bi_rank=args.bi_rank, 
                bi_att_softmax=args.bi_att_softmax, 
                bi_filter_size=args.bi_filter_size, 
                bi_dropout=args.bi_dropout, 
                bi_conv_dropout=args.bi_conv_dropout, 
                dataset=args.dataset)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()
    # print(model)
    # input('...')

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # print(model)
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
            # input('...')
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 10
        # data_length = 5

    if args.train_reverse:
        train_temp_transform = ReverseFrames(size=data_length*args.num_segments)
    elif args.train_shuffle:
        train_temp_transform = ShuffleFrames(size=data_length*args.num_segments)
    else:
        train_temp_transform = IdentityTransform()
    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=img_prefix+"{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   # image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   temp_transform=train_temp_transform, 
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]), 
                   contrastive_mode=args.contrastive_mode),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if args.val_reverse:
        val_temp_transform = ReverseFrames(size=data_length*args.num_segments)
        print('using reverse val')
    elif args.val_shuffle:
        val_temp_transform = ShuffleFrames(size=data_length*args.num_segments)
        print('using shuffle val')
    else:
        val_temp_transform = IdentityTransform()
        print('using normal val')
    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl=img_prefix+"{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   # image_tmpl="image_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   temp_transform=val_temp_transform, 
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        if args.contrastive_mode:
            criterion = ContrastiveLoss(m1=args.contras_m1, 
                                        m2=args.contras_m2).cuda()
            val_criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss().cuda()
            val_criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adagrad(policies,
                                # args.lr,
                                # weight_decay=args.weight_decay)

    if args.evaluate:
        # val_logger = open(os.path.join(args.result_path, 'test.log'), 'w')
        print('evaluating')
        val_logger = os.path.join(args.result_path, 'test.log')
        validate(val_loader, model, val_criterion, 0, val_logger=val_logger)
        # val_logger.close()
        return

    # train_logger = open(os.path.join(args.result_path, 'train.log'), 'w')
    # val_logger = open(os.path.join(args.result_path, 'val.log'), 'w')
    train_logger = os.path.join(args.result_path, 'train.log')
    val_logger = os.path.join(args.result_path, 'val.log')
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        if args.contrastive_mode:
            train_contrastive(train_loader, model, criterion,
                    optimizer, epoch, train_logger=train_logger, args=args)
        else:
            # train for one epoch
            train(train_loader, model, criterion, 
                    optimizer, epoch, train_logger=train_logger)
        # train_logger.write('\n')
        with open(train_logger, 'a') as f:
            f.write('\n')

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, val_criterion, 
                    (epoch + 1) * len(train_loader), val_logger=val_logger)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)
    # train_logger.close()
    # val_logger.close()


def train(train_loader, model, criterion, optimizer, epoch, train_logger):
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
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

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


def validate(val_loader, model, criterion, iter, val_logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            log_line = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            print(log_line)
            # val_logger.write(log_line+'\n')
            with open(val_logger, 'a') as f:
                f.write(log_line+'\n')

    # print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          # .format(top1=top1, top5=top5, loss=losses)))
    log_line = ('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses))
    print(log_line)
    with open(val_logger, 'a') as f:
        f.write(log_line+'\n\n')

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, os.path.join(args.result_path, filename))
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 
            'model_best.pth.tar'))
        with open(os.path.join(args.result_path, 'best_epoch.txt'), 'a') as f:
            f.write('best epoch: '+str(state['epoch']))
        shutil.copyfile(os.path.join(args.result_path, filename), 
                        os.path.join(args.result_path, best_name))


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


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


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


if __name__ == '__main__':
    main()
