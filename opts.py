import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 
                                        'kinetics', 'something'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'lstm', 'conv_lstm', 'ele_multi', 
                        'bilinear_att', 'bilinear_multi_top', 'temp_att_fusion'])
# parser.add_argument('--consensus_type', type=str, default='avg',
                    # choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn'])
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--loss_type', type=str, default="nll",
                    choices=['nll'])
# ====== Modified ======
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
parser.add_argument('--val_reverse', default=False, action='store_true', 
                    help='validate (test) with frames reversed')
parser.add_argument('--val_shuffle', default=False, action='store_true', 
                    help='validate (test) with frames shuffled')
parser.add_argument('--contrastive_mode', default=False, action='store_true', 
                    help='train with contrastive loss')
parser.add_argument('--contras_m1', default=1, type=float, 
                    help='contrastive loss margin 1')
parser.add_argument('--contras_m2', default=1, type=float, 
                    help='contrastive loss margin 2')

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=45, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[20, 40], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--no_partialbn', '--npb', default=False, action="store_true")

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int,
                    metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--result_path', default='result', type=str,
                    metavar='LOG_PATH', help='results and log path')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', default="flow_", type=str)
# parser.add_argument('--flow_prefix', default="", type=str)








