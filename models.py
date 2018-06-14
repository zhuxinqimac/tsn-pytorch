from torch import nn

from ops.basic_ops import ConsensusModule, Identity
from fusion_meth import MeanFusion, MaxFusion, LSTMFusion, EleMultFusion
from fusion_meth import ConvLSTMFusion, BilinearAttentionFusion, BilinearMultiTop
from aBiliP_Module import *
from transforms import *
from torch.nn.init import normal, constant
from conv_lstm import CLSTM_cell

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, inputs):
        return inputs

class TSN(nn.Module):
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True, 
                 lstm_out_type=None, lstm_layers=1, 
                 lstm_hidden_dims=512, conv_lstm_kernel=5, 
                 bi_add_clf=False, bi_out_dims=101, 
                 bi_rank=1, bi_att_softmax=False, bi_filter_size=1, 
                 bi_dropout=0., bi_conv_dropout=0, 
                 get_att_maps=False, dataset='ucf101'):
        super(TSN, self).__init__()
        self.modality = modality
        self.num_segments = num_segments
        self.reshape = True
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.lstm_hidden_dims = lstm_hidden_dims
        self.crop_num = crop_num
        self.consensus_type = consensus_type
        self.bi_out_dims = bi_out_dims
        self.bi_filter_size = bi_filter_size
        self.bi_add_clf = bi_add_clf
        self.bi_rank = bi_rank
        self.bi_att_softmax = bi_att_softmax
        self.bi_dropout = bi_dropout
        self.bi_conv_dropout = bi_conv_dropout
        self.get_att_maps = get_att_maps
        self.dataset = dataset
        self.base_name = base_model
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 10
            # self.new_length = 1 if modality == "RGB" else 5
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))

        self._prepare_base_model(base_model)
        # print(self.base_model)
        # print(list(self.base_model.children()))
        # print(self.base_model.last_layer_name)
        # print('last in features:', getattr(self.base_model, self.base_model.last_layer_name).in_features)

        if self.consensus_type == 'avg' or self.consensus_type == 'max':
            self.feature_dim = self._prepare_tsn(num_class)
            self.late_fusion = True
        elif self.consensus_type == 'lstm' or self.consensus_type == 'ele_multi':
            self.feature_dim = self._prepare_fusion(num_class)
            self.late_fusion = False
        elif self.consensus_type == 'conv_lstm':
            self.feature_shape, self.feature_dim = self._prepare_conv_lstm(num_class)
            self.late_fusion = False
        elif self.consensus_type == 'bilinear_att' or \
            self.consensus_type == 'bilinear_multi_top' or \
            self.consensus_type == 'temp_att_fusion':
            self.feature_dim = self._prepare_bilinear_att(num_class)
            self.late_fusion = False
        # print(self.base_model)
        # print(self.base_model[-1])
        # print(self.base_model.last_layer_name)
        # input('...')

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        # self.consensus = ConsensusModule(consensus_type)
        if self.consensus_type == 'avg':
            self.consensus = MeanFusion()
        elif self.consensus_type == 'max':
            self.consensus = MaxFusion()
        elif self.consensus_type == 'lstm':
            self.lstm_out_type = lstm_out_type
            self.lstm_layers = lstm_layers
            if not self.lstm_out_type:
                ValueError('lstm_out_type should not be None.')
            self.consensus = LSTMFusion(self.feature_dim, 
                            self.lstm_hidden_dims, 
                            self.lstm_out_type, 
                            self.lstm_layers, 
                            self.dropout)
        elif self.consensus_type == 'conv_lstm':
            self.lstm_out_type = lstm_out_type
            self.lstm_layers = lstm_layers
            self.conv_lstm_kernel = conv_lstm_kernel
            if not self.lstm_out_type:
                ValueError('lstm_out_type should not be None.')
            print(self.feature_shape, 
                            self.feature_dim, 
                            self.lstm_hidden_dims, 
                            self.lstm_out_type, 
                            self.lstm_layers, 
                            self.conv_lstm_kernel)
            # input('haha')
            self.consensus = ConvLSTMFusion(
                            self.feature_shape, 
                            self.feature_dim, 
                            self.lstm_hidden_dims, 
                            self.lstm_out_type, 
                            self.lstm_layers, 
                            self.conv_lstm_kernel)
            self.avg_pool_layer = nn.AvgPool2d(self.feature_shape)
        elif self.consensus_type == 'ele_multi':
            self.consensus = EleMultFusion()
        elif self.consensus_type == 'bilinear_att':
            self.consensus = BilinearAttentionFusion(self.feature_dim, 
                            self.bi_out_dims, 
                            self.bi_filter_size, 
                            self.num_segments, 
                            self.bi_rank, 
                            self.bi_att_softmax, 
                            self.bi_dropout, 
                            self.bi_conv_dropout,
                            get_att_maps=self.get_att_maps)
        elif self.consensus_type == 'bilinear_multi_top':
            self.consensus = BilinearMultiTop(self.feature_dim, 
                            num_segments=self.num_segments, 
                            num_class=num_class)
        elif self.consensus_type == 'temp_att_fusion':
            self.consensus = TempAttentionBiFusion(self.num_segments, 
                    self.feature_dim, 
                    out_channel=174, 
                    num_rank=64, to_rank_filter_size=1, 
                    num_cut=64, att_bottleneck=512)
        else:
            ValueError("No such fusion method support: "+consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_bilinear_att(self, num_class):
        if 'resnet' in self.base_name:
            if self.base_name == 'resnet34':
                feature_dim = getattr(self.base_model.layer4[-1].bn2, 'num_features')
            elif self.base_name == 'resnet101':
                feature_dim = getattr(self.base_model.layer4[-1].bn3, 'num_features')
            # print(self.base_model)
            # print(feature_dim)
            # input('...')
            removed = list(self.base_model.children())[:-2]
            self.base_model = nn.Sequential(*removed)
        elif 'vgg' in self.base_name:
            # print(self.base_model)
            feature_dim = getattr(self.base_model.features[-3], 'num_features')
            self.base_model = self.base_model.features
            print(self.base_model)
            print(self.avg_pool)
            # input('...')
        if self.dropout == 0:
            self.new_classifier = nn.Sequential(nn.ReLU(), 
                                    nn.Linear(self.bi_out_dims, num_class))
        else:
            self.new_classifier = nn.Sequential(nn.ReLU(), 
                                    nn.Dropout(p=self.dropout), 
                                    nn.Linear(self.bi_out_dims, num_class))
        std = 0.01
        if self.dropout == 0:
            normal(self.new_classifier[-1].weight, 0, std)
            constant(self.new_classifier[-1].bias, 0)
        else:
            normal(self.new_classifier[-1].weight, 0, std)
            constant(self.new_classifier[-1].bias, 0)
        return feature_dim

    def _prepare_conv_lstm(self, num_class):
        shape = getattr(self.base_model, 'avgpool').kernel_size
        feature_dim = getattr(self.base_model.layer4[-1].bn2, 'num_features')
        removed = list(self.base_model.children())[:-2]
        self.base_model = nn.Sequential(*removed)
        if self.dropout == 0:
            self.new_classifier = nn.Linear(self.lstm_hidden_dims, num_class)
        else:
            self.new_classifier = nn.Sequential(nn.Dropout(p=self.dropout), 
                                nn.Linear(self.lstm_hidden_dims, num_class))
        std = 0.001
        if self.dropout == 0:
            normal(self.new_classifier.weight, 0, std)
            constant(self.new_classifier.bias, 0)
        else:
            normal(self.new_classifier[-1].weight, 0, std)
            constant(self.new_classifier[-1].bias, 0)
        return (shape, shape), feature_dim

    def _prepare_fusion(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        removed = list(self.base_model.children())[:-1]
        self.base_model = nn.Sequential(*removed)
        if self.dropout == 0:
            if self.consensus_type == 'lstm':
                self.new_classifier = nn.Linear(self.lstm_hidden_dims, num_class)
            else:
                self.new_classifier = nn.Linear(feature_dim, num_class)
        else:
            if self.consensus_type == 'lstm':
                self.new_classifier = nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(self.lstm_hidden_dims, num_class))
            else:
                self.new_classifier = nn.Sequential(nn.Dropout(p=self.dropout), 
                                                nn.Linear(feature_dim, num_class))
        std = 0.001
        if self.dropout == 0:
            normal(self.new_classifier.weight, 0, std)
            constant(self.new_classifier.bias, 0)
        else:
            normal(self.new_classifier[-1].weight, 0, std)
            constant(self.new_classifier[-1].bias, 0)
            
        return feature_dim


    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, 
                self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, 
                    nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, 
                    nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)

        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, 
                self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, 
                self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim

    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)
            self.base_model.last_layer_name = 'fc'
            # self.input_size = 224
            if self.dataset == 'something':
                self.input_size = 96 # small
                # self.input_size = 84 # tiny
                if 'resnet' in self.base_name:
                    setattr(self.base_model, 'avgpool', 
                            nn.AvgPool2d(kernel_size=3, stride=1, padding=0, 
                                ceil_mode=False, count_include_pad=True))
                    self.avg_pool = Identity()
                elif 'vgg' in self.base_name:
                    self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, 
                            padding=0, ceil_mode=False, count_include_pad=True)
            else:
                self.input_size = 224
                if 'resnet' in self.base_name:
                    self.avg_pool = Identity()
                elif 'vgg' in self.base_name:
                    self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1, 
                            padding=0, ceil_mode=False, count_include_pad=True)
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            if self.dataset == 'something':
                self.input_size = 96
            else:
                self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]

            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            if self.dataset == 'something':
                self.input_size = 114
            else:
                self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                # print('linear ps: ', ps)
                # input('...')
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.LSTM):
                ps = list(m.parameters())
                # print('lstm ps:', ps)
                # input('...')
                normal_weight.append(ps[0])
                normal_weight.append(ps[1])
                normal_bias.append(ps[2])
                normal_bias.append(ps[3])
            elif isinstance(m, CLSTM_cell):
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                normal_bias.append(ps[1])
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length
        # print(input.size())
        # print(input.view((-1, sample_len) + input.size()[-2:]).size())

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        # print(base_out.size())
        # input('aa')
        if self.consensus_type == 'bilinear_att':
            output = self.bi_att_forward(base_out)
        elif self.consensus_type == 'bilinear_multi_top':
            output = self.bi_multi_top(base_out)
        elif self.consensus_type == 'temp_att_fusion':
            output = self.temp_att_bi(base_out)
        elif self.consensus_type == 'conv_lstm':
            output = self.conv_lstm_forward(base_out)
        elif self.consensus_type == 'lstm':
            output = self.lstm_forward(base_out)
        elif self.consensus_type == 'avg' or self.consensus_type == 'max':
            output = self.late_fusion_forward(base_out)
        elif self.consensus_type == 'ele_multi':
            output = self.ele_multi_forward(base_out)
        # print(output.size())
        # input('...')
        if self.get_att_maps:
            return output
        else:
            return output.squeeze(1)

    def temp_att_bi(self, base_out):
        base_out = self.consensus(base_out)
        return base_out
    
    def bi_multi_top(self, base_out):
        base_out = self.consensus(base_out)
        return base_out

    def bi_att_forward(self, base_out):
        # print(base_out.size())
        # base_out = self.avg_pool(base_out)
        # print(base_out.size())
        # input('hah')
        base_out = self.consensus(base_out)
        if self.bi_add_clf:
            if self.get_att_maps:
                output_results = self.new_classifier(base_out[0])
                output = [output_results, base_out[1], base_out[2], base_out[0]]
            else:
                output = self.new_classifier(base_out)
        else:
            output = base_out
        return output
    def conv_lstm_forward(self, base_out):
        base_out = base_out.view((-1, 
                self.num_segments)+base_out.size()[1:])
        base_out = self.consensus(base_out)
        base_out = self.avg_pool_layer(base_out)
        base_out = base_out.view(base_out.size()[0], -1)
        output = self.new_classifier(base_out)
        return output
    def lstm_forward(self, base_out):
        base_out = base_out.view(base_out.size(0), -1)
        base_out = base_out.view((-1, 
                self.num_segments)+base_out.size()[1:])
        base_out = self.consensus(base_out)
        output = self.new_classifier(base_out)
        return output
    def late_fusion_forward(self, base_out):
        base_out = base_out.view(base_out.size()[0], -1)
        if self.dropout > 0:
            base_out = self.new_fc(base_out)
        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        output = self.consensus(base_out)
        return output
    def ele_multi_forward(self, base_out):
        base_out = base_out.view(base_out.size(0), -1)
        base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
        base_out = self.consensus(base_out)
        output = self.new_classifier(base_out)
        return output

        # # print('before flatten: ', base_out.size())
        # if not self.consensus_type == 'conv_lstm':
            # base_out = base_out.view(base_out.size(0), -1)
        # # print('after flatten: ', base_out.size())

        # if self.late_fusion:
            # if self.dropout > 0:
                # base_out = self.new_fc(base_out)
            # # print('after dropout: ',base_out.size())

            # if not self.before_softmax:
                # base_out = self.softmax(base_out)
            # if self.reshape:
                # base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

            # # print('after reshape: ', base_out.size())
            # output = self.consensus(base_out)
            # # print('after consensus: ', output.size())
        # else:
            # if self.reshape:
                # base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            # # print('after reshape: ', base_out.size())
            # base_out = self.consensus(base_out)
            # # print('after consensus: ', base_out.size())
            # if self.consensus_type == 'conv_lstm':
                # base_out = self.avg_pool_layer(base_out)
                # # print('after avg_pool: ', base_out.size())
                # base_out = base_out.view(base_out.size(0), -1)
            # if not self.consensus_type == 'bilinear_att':
                # output = self.new_classifier(base_out)
            # else:
                # if self.bi_add_clf:
                    # output = self.new_classifier(base_out)
                # else:
                    # output = base_output
            # # print('after classifier: ', output.size())
        # # input('stop...')
        # return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], 
            nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + \
                kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, 
                keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, 
                             conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        # return self.input_size * 256 // 224
        if self.dataset == 'something':
            return self.input_size * 108 // 96
        else:
            return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
