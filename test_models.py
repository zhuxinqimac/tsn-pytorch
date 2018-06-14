import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from temporal_transforms import ReverseFrames, ShuffleFrames
from ops import ConsensusModule
from main import AverageMeter, accuracy

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', \
    'kinetics', 'something'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')
# ==== Modified ====
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'lstm', 'conv_lstm', 'ele_multi', 'bilinear_att'])
parser.add_argument('--lstm_out_type', type=str, help='lstm fusion type', 
                    default='avg', choices=['last', 'max', 'avg'])
parser.add_argument('--lstm_layers', type=int, help='lstm layers', 
                    default=1)
parser.add_argument('--lstm_hidden_dims', type=int, help='lstm hidden dims', 
                    default=512)
parser.add_argument('--conv_lstm_kernel', type=int, help='convlstm kernel size', 
                    default=5)
parser.add_argument('--bi_out_dims', type=int, help='bilinear out dims, should equal to num classes when bi_add_clf is false', 
                    default=101)
parser.add_argument('--bi_rank', type=int, default=1, 
                    help='rank used to approximate bilinear pooling')
parser.add_argument('--bi_att_softmax', default=False, action='store_true', 
                    help='add softmax layer for bilinear attention maps')
parser.add_argument('--bi_filter_size', type=int, default=1, 
                    help='filter size used in bilinear pooling when generating attention maps')
parser.add_argument('--bi_dropout', type=float, default=0, help='dropout used in bilinear pooling')
parser.add_argument('--bi_conv_dropout', type=float, default=0, 
                    help='sep_conv dropout during bilinear att')
parser.add_argument('--bi_add_clf', default=False, action='store_true', 
                    help='add another classifier after bilinear fusion')
parser.add_argument('--train_reverse', default=False, action='store_true', 
                    help='train with frames reversed')
parser.add_argument('--train_shuffle', default=False, action='store_true', 
                    help='train with frames shuffled')
parser.add_argument('--test_reverse', default=False, action='store_true', 
                    help='validate (test) with frames reversed')
parser.add_argument('--test_shuffle', default=False, action='store_true', 
                    help='validate (test) with frames shuffled')
parser.add_argument('--contrastive_mode', default=False, action='store_true', 
                    help='train with contrastive loss')
parser.add_argument('--contras_m1', default=1, type=float, 
                    help='contrastive loss margin 1')
parser.add_argument('--contras_m2', default=1, type=float, 
                    help='contrastive loss margin 2')

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

if not (args.consensus_type == 'lstm' or args.consensus_type == 'conv_lstm'):
    args.lstm_out_type = None
net = TSN(num_class, args.test_segments, args.modality,
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
# net = TSN(num_class, args.test_segments, args.modality,
          # base_model=args.arch,
          # consensus_type=args.crop_fusion_type,
          # dropout=args.dropout)
# net = TSN(num_class, 1, args.modality,
          # base_model=args.arch,
          # consensus_type=args.crop_fusion_type,
          # dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

if args.test_reverse:
    test_temp_transform = ReverseFrames()
elif args.test_shuffle:
    test_temp_transform = ShuffleFrames()
else:
    test_temp_transform = IdentityTransform()

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 10,
                   # new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   # image_tmpl="image_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   image_tmpl=img_prefix+"{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   temp_transform=test_temp_transform, 
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        # length = 10
        length = 20
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    # input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        # volatile=True)
    input_var = torch.autograd.Variable(data.view(-1, args.test_segments, length,
                                    data.size(2), data.size(3)), volatile=True)
    # print(input_var.size())
    # input('...')
    outputs = net(input_var)
    rst = outputs.data.cpu().numpy().copy()
    # print(rst.size)
    # input('...')
    target = label.cuda(async=True)
    # prec1, prec5 = accuracy(outputs.view((num_crop, num_class)).mean(dim=0, keepdim=True), 
            # target, topk=(1,5))
    prec1, prec5 = accuracy(outputs.data, target, topk=(1,5))
    top1.update(prec1[0], 1)
    top5.update(prec5[0], 1)
    # return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        # (args.test_segments, 1, num_class)
    # ), label[0]
    # rst = net(input_var).data.cpu().numpy().copy()
    # print(rst.shape)
    # # input('...')
    return i, rst.reshape((num_crop, num_class)).mean(axis=0), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

top1 = AverageMeter()
top5 = AverageMeter()
for i, (data, label) in data_gen:
    if i >= max_num:
        break
    # print(data.size())
    # print(label.size())
    # input('...')
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'
          'Prec@1 {top1.val:.3f} ({top1.avg:.3f}) '
          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(i, i+1,
                            total_num,
                            float(cnt_time) / (i+1), 
                            top1=top1, top5=top5))

# video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]
# print(output[:4])
video_pred = [np.argmax(x[0]) for x in output]
# print(video_pred)
video_labels = [x[1] for x in output]
# print(video_labels)

cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


