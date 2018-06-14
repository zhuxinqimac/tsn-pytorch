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
from visualization import *

best_prec1 = 0


def main():
    global args, best_prec1, class_to_name
    parser.add_argument('--class_index', type=str, help='class index file')
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
        with open(args.class_index, 'r') as f:
            content = f.readlines()
        class_to_name = {idx:line.strip().replace(' ', '-') for idx, line in enumerate(content)}
    else:
        img_prefix = 'image_'
        with open(args.class_index, 'r') as f:
            content = f.readlines()
        class_to_name = {int(line.strip().split(' ')[0])-1:line.strip().split(' ')[1] \
                for line in content}

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
                get_att_maps=True, 
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
        rev_normalize = ReverseGroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 10
        # data_length = 5

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
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # val_logger = open(os.path.join(args.result_path, 'test.log'), 'w')
    print('visualizing...')
    val_logger = os.path.join(args.result_path, 'visualize.log')
    validate(val_loader, model, 0, val_logger=val_logger, rev_normalize=rev_normalize)
    return



def validate(val_loader, model, iters, val_logger, rev_normalize):
    # switch to evaluate mode
    model.eval()

    (inputs, target) = next(iter(val_loader)) 
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(inputs, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    print(input_var.size())

    # output: [batch x 174]
    # sep_maps: [batch x 8 x 256 x 49]
    # join_maps: [batch x 256 x 49]
    # output = model(input_var)
    # print(output)
    input('..')
    output, sep_maps, join_maps, _ = model(input_var)
    
    sep_maps = sep_maps.view(sep_maps.size()[:-1]+\
            (int(np.sqrt(sep_maps.size()[-1])), 
                int(np.sqrt(sep_maps.size()[-1]))))

    # print(inputs.size()) # (2,24,224,224)
    inputs = inputs.view((-1,)+inputs.size()[-2:]) # (48,224,224)
    inputs = rev_normalize(inputs)
    inputs = inputs.view((args.batch_size, args.num_segments, -1)+\
            inputs.size()[-2:])
    np_inputs = inputs.cpu().numpy().copy() 
    np_inputs = np_inputs.transpose((0,1,3,4,2)) # (2,8,224,224,3)
    np_sep_maps = sep_maps.data.cpu().numpy().copy()
    
    map_slice = np_sep_maps[:,:,0,...] # (2,8,7,7)

#=====show single video's multiple ranks ===
    # map_slice = np_sep_maps[0,:,:args.batch_size,...].transpose((1,0,2,3)) # (2,8,7,7)
    # for i in range(1, np_inputs.shape[0]):
        # np_inputs[i] = np_inputs[0]
#=====show single video's multiple ranks ===

    atts_resized = []
    for i in range(map_slice.shape[0]):
        tmp = []
        for j in range(map_slice.shape[1]):
            tmp.append(scipy.misc.imresize(map_slice[i,j], (np_inputs.shape[-3], 
                                                            np_inputs.shape[-2])))
        # print(tmp)
        atts_resized.append(tmp)
    atts_resized = np.array(atts_resized)/255.
    print(atts_resized.mean(axis=-1).mean(axis=-1))
    atts_resized = np.clip(atts_resized+0.3, 0., 1.) # for better visualization
    atts_resized = np.expand_dims(atts_resized, axis=-1)
    atts_resized = np.repeat(atts_resized, 3, axis=-1)
    # print(np_inputs)
    # input('..')

    np_target = target.cpu().numpy().copy()
    video_names = [class_to_name[i] for i in np_target[:args.batch_size]]
    show_frames_att(np_inputs, atts_resized, video_names, img_name='./imgs/nn.pdf')
    return 


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
