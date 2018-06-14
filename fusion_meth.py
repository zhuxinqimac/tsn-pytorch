import torch
import torch.autograd as autograd
import torch.nn as nn
from conv_lstm import CLSTM
from conv_lstm import weights_init
from torch.nn.init import normal, constant
from aBiliP_Module import ABP_Video

class MeanFusion(torch.nn.Module):
    def __init__(self, dim=1):
        super(MeanFusion, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        inputs = inputs.mean(dim=self.dim)
        return inputs

class MaxFusion(torch.nn.Module):
    def __init__(self, dim=1):
        super(MaxFusion, self).__init__()
        self.dim = dim
    def forward(self, inputs):
        inputs, _ = inputs.max(dim=self.dim)
        return inputs


class EleMultFusion(torch.nn.Module):
    def __init__(self, dim=1):
        super(EleMultFusion, self).__init__()
        self.dim = dim
    def forward(self, inputs):
        multi = inputs[:, 0, :]
        for i in range(1, inputs.size(1)):
            multi = multi * inputs[:, i, :]
        return multi


class LSTMFusion_segment_fault(torch.nn.Module):
    def __init__(self, feature_in=512, feature_out=512, 
                out_type='last', lstm_layers=1, dropout=0.8, dim=1):
        super(LSTMFusion, self).__init__()
        self.dim = dim
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.out_type = out_type
        self.lstm = nn.LSTM(self.feature_in, self.feature_out, self.lstm_layers)
        # self.lstm = nn.LSTM(self.feature_in, self.feature_out, self.lstm_layers,
                        # dropout=self.dropout)
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 0, 1)
        hdims = inputs.size()
        # print(hdims)
        hidden = (autograd.Variable(torch.randn(self.lstm_layers, hdims[1], self.feature_out).cuda()), 
            autograd.Variable(torch.randn(self.lstm_layers, hdims[1], self.feature_out).cuda()))
        self.lstm.flatten_parameters()
        seq_out, last_out_bi = self.lstm(inputs, hidden)
        # print('seq_out:', seq_out.size())
        # print('last_out_bi[0]: ', last_out_bi[0].size())

        if self.out_type == 'last':
            # output = torch.transpose(last_out_bi[0], 0, 1)
            output = seq_out[-1]
        elif self.out_type == 'max':
            output = torch.transpose(seq_out, 0, 1)
            output, _ = output.max(dim=self.dim)
        else:
            output = torch.transpose(seq_out, 0, 1)
            output = output.mean(dim=self.dim)
        return output


class LSTMFusion(torch.nn.Module):
    def __init__(self, feature_in=512, feature_out=512, 
                    out_type='last', lstm_layers=1, dropout=0., dim=1):
        super(LSTMFusion, self).__init__()
        self.dim = dim
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.lstm_layers = lstm_layers
        self.out_type = out_type
        print(self.feature_out)
        print(self.feature_in)
        self.conv_lstm = CLSTM((1,1), self.feature_in, 
                    1, self.feature_out, 
                    self.lstm_layers)
        self.conv_lstm.apply(weights_init)
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 0, 1)
        inputs = inputs.unsqueeze(-1)
        inputs = inputs.unsqueeze(-1)
        # print('inputs size:', inputs.size())
        # input('...')
        hdims = inputs.size()

        hidden_state = self.conv_lstm.init_hidden(inputs.size()[1])
        _, seq_out = self.conv_lstm(inputs, hidden_state)
        seq_out = seq_out.squeeze(-1)
        seq_out = seq_out.squeeze(-1)
        # print(seq_out.size())
        # input('...')
        # print('seq_out:', seq_out.size())
        # print('last_out_bi[0]: ', last_out_bi[0].size())

        if self.out_type == 'last':
            output = seq_out[-1]
        elif self.out_type == 'max':
            output = torch.transpose(seq_out, 0, 1)
            output, _ = output.max(dim=self.dim)
        else:
            output = torch.transpose(seq_out, 0, 1)
            output = output.mean(dim=self.dim)
        # print(output.size())
        # input('...')
        return output

class ConvLSTMFusion(torch.nn.Module):
    def __init__(self, shape, feature_in=512, feature_out=512, 
                    out_type='last', lstm_layers=1, filter_size=5, dim=1):
        super(ConvLSTMFusion, self).__init__()
        self.dim = dim
        self.shape = shape
        self.feature_in = feature_in
        self.filter_size = filter_size
        self.feature_out = feature_out
        self.lstm_layers = lstm_layers
        self.out_type = out_type
        self.conv_lstm = CLSTM(self.shape, self.feature_in, 
                    self.filter_size, self.feature_out, 
                    self.lstm_layers)
        self.conv_lstm.apply(weights_init)
    def forward(self, inputs):
        inputs = torch.transpose(inputs, 0, 1)
        # print('conv lstm inputs size:', inputs.size())
        # input('...')
        hdims = inputs.size()

        hidden_state = self.conv_lstm.init_hidden(inputs.size()[1])
        _, seq_out = self.conv_lstm(inputs, hidden_state)
        # print('seq_out:', seq_out.size())
        # print('last_out_bi[0]: ', last_out_bi[0].size())

        if self.out_type == 'last':
            output = seq_out[-1]
        elif self.out_type == 'max':
            output = torch.transpose(seq_out, 0, 1)
            output, _ = output.max(dim=self.dim)
        else:
            output = torch.transpose(seq_out, 0, 1)
            output = output.mean(dim=self.dim)
        # print(output.size())
        # input('...')
        return output

class BilinearAttentionFusion_false(torch.nn.Module):
    def __init__(self, feature_in=512, feature_out=101,
                filter_size=1, num_segments=3, rank=1, dim=1):
        super(BilinearAttentionFusion, self).__init__()
        self.dim = dim
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.filter_size = filter_size
        self.num_segments = num_segments
        self.rank = rank
        self.sep_conv = nn.Conv2d(self.num_segments*self.feature_in, 
                self.num_segments*self.feature_out, 
                self.filter_size, 
                groups=self.num_segments)
    def forward(self, inputs):
        inputs = inputs.view(-1, self.num_segments*self.feature_in, 
                    inputs.size()[-2:])
        output = autograd.Variable(torch.zeros(inputs.size()[0], 
                                            self.feature_out))
        for i in range(self.rank):
            x = self.sep_conv(inputs)
            x = x.view(x.size()[0], self.num_segments, 
                    self.feature_out, x.size()[-2]*x.size()[-1])
            y = x[:, 0, ...]
            for i in range(1, self.num_segments):
                y = x[:, i, ...]*y
            print('before mean:', y.size())
            y = y.mean(dim=-1)
            print('after mean:', y.size())
            input('...')
            output = output+y
        print(output.size())
        input('...')
        return output

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, inputs):
        return inputs

class BilinearAttentionFusion(torch.nn.Module):
    def __init__(self, feature_in=512, feature_out=101,
                filter_size=1, num_segments=3, rank=1, 
                bi_att_softmax=False, dropout=0, bi_conv_dropout=0, 
                get_att_maps=False, dim=1):
        super(BilinearAttentionFusion, self).__init__()
        self.dim = dim
        self.feature_in = feature_in
        self.feature_out = feature_out
        self.filter_size = filter_size
        self.num_segments = num_segments
        self.rank = rank
        self.dropout = dropout
        self.bi_att_softmax = bi_att_softmax
        self.bi_conv_dropout = bi_conv_dropout
        self.get_att_maps = get_att_maps

        std = 0.01
        self.sep_conv = nn.Conv2d(self.num_segments*self.feature_in, 
                self.num_segments*self.rank, 
                self.filter_size, 
                groups=self.num_segments)
        normal(self.sep_conv.weight, 0, std)
        constant(self.sep_conv.bias, 0)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        if self.bi_conv_dropout>0:
            print('using conv dropout')
            self.bi_conv_dropout = nn.Dropout2d(p=self.bi_conv_dropout)
        else:
            self.bi_conv_dropout = Identity()

        if self.bi_att_softmax:
            self.softmax = nn.Softmax(dim=2)
        else:
            self.softmax = Identity()

        if self.dropout>0:
            self.linear = nn.Sequential(nn.Dropout(p=self.dropout), 
                                nn.Linear(self.rank, self.feature_out))
            normal(self.linear[-1].weight, 0, std)
            constant(self.linear[-1].bias, 0)
        else:
            self.linear = nn.Linear(self.rank, self.feature_out)
            normal(self.linear.weight, 0, std)
            constant(self.linear.bias, 0)

    def forward(self, inputs):
        inputs = inputs.view((-1, self.num_segments*self.feature_in) + \
                    inputs.size()[-2:])
        # print('inputs size:', inputs.size())
        # input('ll')
        x = self.sep_conv(inputs)
        x = self.bi_conv_dropout(x)
        # x = self.relu(x)
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        x_size = x.size()
        # print('x size:', x.size())
        x = x.view(x_size[0], self.num_segments, 
                    self.rank, x_size[-2]*x_size[-1])
        x_softmax = x.transpose(1,2).contiguous()
        x_softmax = x_softmax.view(x_size[0], 
                self.rank, self.num_segments*x_size[-2]*x_size[-1])
        # print('before softmax:', x.size())
        # x_sig = self.sigmoid(x)
        # x_rel = self.relu(x)
        x_softmax = self.softmax(x_softmax)
        x_softmax = x_softmax.view(x_size[0], 
                self.rank, self.num_segments, x_size[-2]*x_size[-1])
        x_softmax = x_softmax.transpose(1,2).contiguous()
        # print('after softmax:', x.size())
        # print(torch.sum(x, -1)[0, 0])
        # input('...')
        #=== 
        # y = x[:, 0, ...]
        # for i in range(1, self.num_segments):
            # y = x[:, i, ...]*y
        #=== 
        # y = x[:, 0, ...]*x[:, 1, ...]
        # for i in range(1, self.num_segments//2):
            # y = x[:, i*2, ...]*x[:, i*2+1, ...]+y
        #=== 
        # y = x[:, 0, ...]*x[:, 1, ...]
        # for i in range(1, self.num_segments-1):
            # y = x[:, i, ...]*x[:, i+1, ...]+y
        #===
        # y1 = x[:, :-1, ...]
        # y2 = x[:, 1:, ...]
        # y = y1*y2
        # y = y.sum(dim=1)
        #===
        y_map = x[:, 0, ...]*x[:, 1, ...]
        for i in range(2, self.num_segments):
            y_map = x[:, 0, ...]*x[:, i, ...]+y_map
        #===
        # y = x_rel[:, 0, ...]*x_sig[:, 1, ...]
        # for i in range(2, self.num_segments):
            # y = x_rel[:, 0, ...]*x_sig[:, i, ...]+y
        #===
        # print('before mean:', y.size())
        y = y_map.mean(dim=-1)
        # print('after mean:', y.size())
        # input('..')
        output = self.linear(y)
        # print('output size:', output.size())
        # input('...')
        if self.get_att_maps:
            return output, x_softmax, y_map
        else:
            return output
        
class BilinearMultiTop(torch.nn.Module):
    def __init__(self, feature_dim, 
                num_segments, num_class, 
        # print(inputs.size())
        # input('lll')
        # print(inputs.size())
        # input('lll')
                num_layer=1, rank_list=[1024, 1024], 
                out_channel_list=[1024, 1024], 
                pool_filter_list=[3, 1], 
                pool_stride_list=[1, 1], 
                final_dropout=0.):
        super(BilinearMultiTop, self).__init__()
        self.tmp_in_dim = feature_dim
        self.tmp_num_segments = num_segments
        self.out_size = 1
        self.bi_layers = []
        self.num_class = num_class
        self.num_layer = num_layer
        self.final_dropout = final_dropout
        for i in range(self.num_layer):
            tmp_layer = ABP_Video(self.tmp_num_segments, 
                    self.tmp_in_dim, out_channel_list[i], 
                    pool_filter_size=pool_filter_list[i], 
                    pool_stride=pool_stride_list[i], 
                    num_rank=rank_list[i])
            self.bi_layers.append(tmp_layer)
            self.bi_layers.append(nn.ReLU())

            self.bi_layers.append(nn.Conv2d(out_channel_list[i], 
                    out_channel_list[i], 
                    1))

            self.tmp_num_segments -= 1
            # self.tmp_num_segments = self.tmp_num_segments//2
            self.tmp_in_dim = out_channel_list[i]

        self.layers = nn.Sequential(*self.bi_layers)
        self.final_avg_pool = nn.AvgPool2d(kernel_size=self.out_size, stride=1)
        self.final_num_segments = self.tmp_num_segments
        self.final_out_channdel = out_channel_list[-1]

        std = 0.01
        if self.final_dropout>0.:
            self.final_linear = nn.Sequential(nn.Dropout(p=self.final_dropout), 
                                    nn.Linear(self.final_out_channdel,
                                        self.num_class))
            normal(self.final_linear[-1].weight, 0, std)
            constant(self.final_linear[-1].bias, 0)
        else:
            self.final_linear = nn.Linear(self.final_out_channdel, 
                                        self.num_class)
            normal(self.final_linear.weight, 0, std)
            constant(self.final_linear.bias, 0)
        
    def forward(self, inputs):
        '''
        in_feature: (batch x num_seg, in_channel, h, w)
        out_feature: (batch, num_class)
        '''
        inputs = self.layers(inputs)
        # print(inputs.size())
        # input('lll')

        inputs = self.final_avg_pool(inputs)

        inputs = inputs.view((-1, self.final_num_segments, 
            self.final_out_channdel))
        inputs = inputs.mean(dim=1)

        inputs = self.final_linear(inputs)
        
        return inputs

