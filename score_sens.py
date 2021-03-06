import argparse
import torch
import os
import time
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import numpy as np

from dataset import TSNDataSet
from models import TSN
from transforms import *
from temporal_transforms import ReverseFrames, ShuffleFrames
from main import AverageMeter, accuracy

def evaluate(test_loader, model, criterion, eval_logger, softmax, 
        analysis_recorder):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    n_s_top1 = AverageMeter()
    n_s_top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (inputs, target) in enumerate(test_loader):
        # print(inputs[0].size())
        # input('...')
        target = target.cuda(async=True)
        norm_input_var = torch.autograd.Variable(inputs[0], volatile=True)
        abnorm_input_var = torch.autograd.Variable(inputs[1], volatile=True)
        v_path = inputs[4][0].replace(' ', '-')
        # idx_list = inputs[2]
        # t_to_idx = {x:i for i, x in enumerate(idx_list[0])}
        # ab_idx_list = inputs[3]
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        norm_output = model(norm_input_var)
# === garbage ===
        # norm_output_ab = torch.autograd.Variable(torch.Tensor(norm_output.size())).cuda()
        # # print(norm_output.size()) # [1, 16, 512, 7, 7]
        # for j in range(norm_output.size()[1]):
            # # print(idx_list[0][j])
            # norm_output_ab[0, j, ...] = norm_output[0, t_to_idx[ab_idx_list[0, j]], ...]
        # # print(norm_output_ab[0][0, 0, ...])
# === garbage ===
        abnorm_output = model(abnorm_input_var)
        # print(norm_output)
        # print(norm_output.size())
        # print(abnorm_output)
        # print(abnorm_output.size())
        # input('..')
        # loss = criterion(output, target_var)
        norm_sm = softmax(norm_output)
        abnorm_sm = softmax(abnorm_output)
        loss = criterion(norm_sm, abnorm_sm)
        # print('np_loss:', np_loss)
        loss = torch.sqrt(loss)
        # print('loss:', loss)
        # input('...')
        prec1, prec5 = accuracy(norm_output.data, target, topk=(1,5))
        top1.update(prec1[0], 1)
        top5.update(prec5[0], 1)
        prec1, prec5 = accuracy(abnorm_output.data, target, topk=(1,5))
        n_s_top1.update(prec1[0], 1)
        n_s_top5.update(prec5[0], 1)

        _, n_n_pred = norm_sm.max(1)
        _, n_s_pred = abnorm_sm.max(1)
        GT_class_name = class_to_name[target.cpu().numpy()[0]]
        # print(norm_sm)
        # print('v_path:', v_path)
        # print('n_n_pred:', n_n_pred)
        # print('n_s_pred:', n_s_pred)
        # print('target:', target)
        # print('GT_class_name:', GT_class_name)
        if (n_n_pred.data == target).cpu().numpy():
            if_correct = 1
        else:
            if_correct = 0
        # print('if_correct:', if_correct)
        # input('...')


        losses.update(loss.data[0], 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        analysis_data_line = ('{path} {if_correct} {loss.val:.4f} '
                '{GT_class_name} {GT_class_index} '
                '{n_n_pred} {n_s_pred}'.format(
                    path=v_path, if_correct=if_correct, loss=losses, 
                    GT_class_name=GT_class_name, 
                    GT_class_index=target.cpu().numpy()[0], 
                    n_n_pred=n_n_pred.data.cpu().numpy()[0], 
                    n_s_pred=n_s_pred.data.cpu().numpy()[0]))
        with open(analysis_recorder, 'a') as f:
            f.write(analysis_data_line+'\n')

        if i % 20 == 0:
            log_line = ('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'n_s_Prec@1 {n_s_top1.val:.3f} ({n_s_top1.avg:.3f})\t'
                  'n_s_Prec@5 {n_s_top5.val:.3f} ({n_s_top5.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses, 
                   top1=top1, top5=top5, n_s_top1=n_s_top1, n_s_top5=n_s_top5))
            print(log_line)
            # eval_logger.write(log_line+'\n')
            with open(eval_logger, 'a') as f:
                f.write(log_line+'\n')

    # print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          # .format(top1=top1, top5=top5, loss=losses)))
    log_line = ('Testing Results: Loss {loss.avg:.5f}'
          .format(loss=losses))
    print(log_line)
    with open(eval_logger, 'a') as f:
        f.write(log_line+'\n\n')

    return


class FeatureMapModel(torch.nn.Module):
    def __init__(self, whole_model, consensus_type, modality, num_segments):
        super(FeatureMapModel, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        if self.modality == 'RGB':
            self.new_length = 1
        else:
            self.new_length = 10
        if consensus_type == 'bilinear_att' or consensus_type == 'conv_lstm':
            self.base_model = whole_model.module.base_model
        elif consensus_type == 'lstm' or consensus_type == 'ele_multi':
            removed = list(whole_model.module.base_model.children())[:-1]
            self.base_model = torch.nn.Sequential(*removed)
        elif consensus_type == 'avg' or consensus_type == 'max':
            removed = list(whole_model.module.base_model.children())[:-2]
            self.base_model = torch.nn.Sequential(*removed)
        else:
            ValueError(('Not supported consensus \
                        type {}.'.format(self.consensus_type)))
        # print(self.base_model)

    def forward(self, inputs):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # print(input.size())
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())

        base_out = self.base_model(inputs.view((-1, 
                        sample_len) + inputs.size()[-2:]))
        base_out = base_out.view((-1, self.num_segments) + \
                        base_out.size()[-3:])
        # base_out = base_out.mean(dim=1)
        # return base_out.squeeze(1)
        return base_out



if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
                "Get a model's sensitivity to temporal info.")
    parser.add_argument('model_path', type=str, 
                help='Path to a pretrained model.')
    parser.add_argument('consensus_type', type=str, 
                help='Consensus type.', 
                choices=['avg', 'max', 'lstm', 
                    'conv_lstm', 'ele_multi', 'bilinear_att'])
    parser.add_argument('dataset', type=str, 
                choices=['ucf101', 'something'])
    parser.add_argument('test_list', type=str, help='Test list.')
    parser.add_argument('result_path', default='result', type=str,
                        metavar='LOG_PATH', help='results and log path')
    parser.add_argument('class_index', type=str, help='class index file')
    parser.add_argument('--num_segments', type=int, default=3)
    parser.add_argument('--modality', type=str, default='RGB', 
                choices=['RGB', 'Flow'])
    parser.add_argument('--arch', type=str, default="resnet34")

    # ====== Modified ======
    parser.add_argument('--lstm_out_type', type=str, help='lstm fusion type', 
                        default='avg', choices=['last', 'max', 'avg'])
    parser.add_argument('--lstm_layers', type=int, help='lstm layers', 
                        default=1)
    parser.add_argument('--lstm_hidden_dims', type=int, help='lstm hidden dims', 
                        default=512)
    parser.add_argument('--conv_lstm_kernel', type=int, 
                        help='convlstm kernel size', default=5)
    parser.add_argument('--flow_prefix', default="", type=str)
    # parser.add_argument('--scale_size', default=256, type=int, 
                        # help='size to be scaled before crop (default 256)')
    # parser.add_argument('--crop_size', default=224, type=int, 
                        # help='size to be cropped to (default 224)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--compared_temp_transform', default='shuffle', 
                        type=str, help='temp transform to compare', 
                        choices=['shuffle', 'reverse'])
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')
    parser.add_argument('--bi_out_dims', type=int, help='bilinear out dims, should equal to num classes when bi_add_clf is false', 
                        default=101)
    parser.add_argument('--bi_rank', type=int, default=1, 
                        help='rank used to approximate bilinear pooling')
    parser.add_argument('--bi_filter_size', type=int, default=1, 
                        help='filter size used in bilinear pooling when generating attention maps')
    parser.add_argument('--bi_dropout', type=float, default=0, help='dropout used in bilinear pooling')
    parser.add_argument('--bi_add_clf', default=False, action='store_true', 
                        help='add another classifier after bilinear fusion')
    parser.add_argument('--bi_conv_dropout', type=float, default=0, 
                        help='sep_conv dropout during bilinear att')
    parser.add_argument('--bi_att_softmax', default=False, action='store_true', 
                        help='add softmax layer for bilinear attention maps')

    
    args = parser.parse_args()
    if args.dataset == 'ucf101':
        num_class = 101
        img_prefix = 'image_'
        with open(args.class_index, 'r') as f:
            content = f.readlines()
        class_to_name = {int(line.strip().split(' ')[0])-1:line.strip().split(' ')[1] \
                for line in content}
    else:
        num_class = 174
        img_prefix = ''
        with open(args.class_index, 'r') as f:
            content = f.readlines()
        class_to_name = {idx:line.strip().replace(' ', '-') for idx, line in enumerate(content)}
    
    whole_model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, 
                dropout=args.dropout, 
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

    crop_size = whole_model.crop_size
    scale_size = whole_model.scale_size
    input_mean = whole_model.input_mean
    input_std = whole_model.input_std

    whole_model = torch.nn.DataParallel(whole_model, 
                device_ids=args.gpus).cuda()

    if os.path.isfile(args.model_path):
        print(("=> loading checkpoint '{}'".format(args.model_path)))
        checkpoint = torch.load(args.model_path)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        # print(whole_model)
        whole_model.load_state_dict(checkpoint['state_dict'])
        # print(("=> loaded checkpoint epoch {}"
              # .format(checkpoint['epoch'])))
    else:
        ValueError(('No check point found at "{}"'.format(args.model_path)))

    # model = FeatureMapModel(whole_model, args.consensus_type, 
                        # args.modality, args.num_segments)
    model = whole_model
    # input('...')

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    # model = torch.nn.DataParallel(model.cuda(devices[0]), device_ids=devices)

    normalize = GroupNormalize(input_mean, input_std)

    if args.compared_temp_transform == 'shuffle':
        temp_transform = ShuffleFrames()
    else:
        temp_transform = ReverseFrames()

    test_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.num_segments,
                   new_length=1 if args.modality == "RGB" else 10,
                   modality=args.modality,
                   image_tmpl=img_prefix+"{:05d}.jpg" if args.modality \
                           in ["RGB", "RGBDiff"] else \
                           args.flow_prefix+"{}_{:05d}.jpg",
                   # image_tmpl="image_{:05d}.jpg" if args.modality \
                           # in ['RGB', 'RGBDiff'] else \
                           # args.flow_prefix+"{}_{:05d}.jpg",
                   temp_transform=temp_transform, 
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ]), 
                   score_sens_mode=True),
        batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cudnn.benchmark = True

    # if args.measure_type == 'KL':
        # criterion = torch.nn.KLDivLoss().cuda()
    # else:
        # criterion = torch.nn.MSELoss().cuda()
    criterion = torch.nn.MSELoss().cuda()
    softmax = torch.nn.Softmax(1).cuda()

    eval_logger = os.path.join(args.result_path, 'sens_log.log')
    with open(eval_logger, 'w') as f:
        f.write('')
    analysis_recorder = os.path.join(args.result_path, 'sens_analysis.txt')
    with open(analysis_recorder, 'w') as f:
        f.write('')
    evaluate(test_loader, model, criterion, eval_logger=eval_logger, 
            softmax=softmax, analysis_recorder=analysis_recorder)
    
