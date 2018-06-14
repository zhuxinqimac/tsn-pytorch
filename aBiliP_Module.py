import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np

from torch.nn.init import normal, constant


class ABP_Video(torch.nn.Module):
    def __init__(self, num_segment, in_channel, out_channel, 
            pool_filter_size=2, pool_stride=2, num_rank=64, 
            to_rank_filter_size=1, 
            out_filter_size=1, to_out_conv_dropout=0.):
        super(ABP_Video, self).__init__()
        self.num_segment = num_segment
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool_filter_size = pool_filter_size
        self.pool_stride = pool_stride
        self.num_rank = num_rank
        self.to_rank_filter_size = to_rank_filter_size
        self.out_filter_size = out_filter_size
        self.to_out_conv_dropout = to_out_conv_dropout

        self.to_rank_conv = nn.Conv2d(self.num_segment*self.in_channel, 
                self.num_segment*self.num_rank, 
                self.to_rank_filter_size, 
                groups=self.num_segment)
        
        self.avg_pool = nn.AvgPool2d(self.pool_filter_size, 
                self.pool_stride)

        if self.to_out_conv_dropout > 0.:
            self.to_out_conv = nn.Sequential(nn.Dropout2d(
                p=self.to_out_conv_dropout), 
                nn.Conv2d((self.num_segment-1)*self.num_rank, 
                    (self.num_segment-1)*self.out_channel, 
                    self.out_filter_size, 
                    groups=(self.num_segment-1)))
            self.to_out_conv_shared = nn.Sequential(nn.Dropout2d(
                p=self.to_out_conv_dropout), 
                nn.Conv2d(self.num_rank, 
                    self.out_channel, 
                    self.out_filter_size))
        else:
            self.to_out_conv = nn.Conv2d((self.num_segment-1)*self.num_rank, 
                    (self.num_segment-1)*self.out_channel, 
                    self.out_filter_size, 
                    groups=(self.num_segment-1))
            self.to_out_conv_shared = nn.Conv2d(self.num_rank, 
                    self.out_channel, 
                    self.out_filter_size)
        self.to_out_linear = nn.Linear(self.num_rank, self.out_channel)

    def forward(self, inputs):
        '''
        in_feature: (batch x num_seg, in_channel, h, w)
        '''
        inputs = inputs.view((-1, self.num_segment*self.in_channel)+\
                inputs.size()[-2:])
        inputs = self.to_rank_conv(inputs)
        inputs = inputs.view((-1, self.num_segment, self.num_rank)+\
                inputs.size()[-2:])
        inputs_former = inputs[:, :-1, ...]
        inputs_latter = inputs[:, 1:, ...]
        # inputs_former = inputs[:, list(range(0, self.num_segment, 2)), ...]
        # inputs_latter = inputs[:, list(range(1, self.num_segment, 2)), ...]
        inputs = inputs_former*inputs_latter
        
        inputs = inputs.view((-1, self.num_rank)+\
                inputs.size()[-2:])
        inputs = self.avg_pool(inputs)
# === None shared conv
        # inputs = inputs.view((-1, (self.num_segment-1)*self.num_rank)+\
                # inputs.size()[-2:])
        # inputs = self.to_out_conv(inputs)
        # inputs = inputs.view((-1, self.out_channel)+\
                # inputs.size()[-2:])
# === None shared conv
# === Shared conv
        inputs = self.to_out_conv_shared(inputs)
# === Shared conv
        # print(inputs.size())
        # input('oo')
        return inputs

class TempAttentionBiFusion(torch.nn.Module):
    def __init__(self, num_segment, in_channel, out_channel, 
            num_rank=64, to_rank_filter_size=1, 
            num_cut=32, att_bottleneck=512):
        super(TempAttentionBiFusion, self).__init__()
        self.num_segment = num_segment
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_rank = num_rank
        self.to_rank_filter_size = to_rank_filter_size
        self.num_cut = num_cut
        self.att_bottleneck = att_bottleneck

        self.temp_att_net = TempAttNet(self.num_segment, num_cut=self.num_cut, 
                num_rank=self.num_rank, att_bottleneck=self.att_bottleneck)


        self.to_rank_conv = nn.Conv2d(self.num_segment*self.in_channel, 
                self.num_segment*self.num_rank, 
                self.to_rank_filter_size, 
                groups=self.num_segment)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.to_out_linear = nn.Linear(self.num_rank, self.out_channel)

    def forward(self, inputs):
        '''
        in_feature: (batch x num_seg, in_channel, h, w)
        out_feature: (batch, out_channel)
        '''
        temp_att = self.temp_att_net(inputs)
        _, max_indices = temp_att.max(1)
        temp_att = temp_att.view(temp_att.size()+(1,1))

        inputs = inputs.view((-1, self.num_segment*self.in_channel)+\
                inputs.size()[-2:])
        inputs = self.to_rank_conv(inputs)
        inputs = inputs.view((-1, self.num_segment, self.num_rank)+\
                inputs.size()[-2:])

        temp_att = temp_att.repeat(1,1,1,inputs.size()[-2], inputs.size()[-1])
        inputs = inputs*temp_att

        #=== efficient
        ta_size = temp_att.size()
        dim0 = np.array(list(range(ta_size[0])), 
                dtype=np.int).repeat(ta_size[2])
        dim1 = max_indices.view(-1)
        dim2 = np.array(list(range(ta_size[2])), 
                dtype=np.int).reshape(1, ta_size[2]).repeat(
                        ta_size[0], axis=0).reshape(-1)
        inputs_pivot = inputs[dim0, dim1, dim2].view((
            ta_size[0], 1, ta_size[2], ta_size[3], ta_size[4])).repeat(
                    1, inputs.size()[1], 1, 1, 1)
        inputs[dim0, dim1, dim2] = 0.

        inputs = inputs_pivot*inputs
        inputs = inputs.sum(1)
        inputs = inputs/(self.num_segment-1)
        #=== efficient

        #=== old
        # inputs_pivot = torch.empty((inputs.size()[0], inputs.size(1)-1)+\
                # inputs.size()[2:], device='cuda:0')
        # inputs_rest = torch.empty((inputs.size()[0], inputs.size(1)-1)+\
                # inputs.size()[2:], device='cuda:0')
        # for i in range(inputs.size()[0]):
            # for j in range(inputs.size()[2]):
                # inputs_pivot[i, :, j, ...] = inputs[i, max_indices[i, j], 
                        # j, ...]
                # inputs_rest[i, :, j, ...] = inputs[i, 
                        # list(range(max_indices[i,j]))+\
                                # list(range(max_indices[i,j]+1, self.num_segment)), 
                                # j, ...]
        # inputs = inputs_pivot*inputs_rest
        # inputs = inputs.mean(1)
        #=== old
        
        
        # inputs = inputs.view((-1, self.num_rank)+\
                # inputs.size()[-2:])
        inputs = self.avg_pool(inputs)
        inputs = inputs.view((-1, self.num_rank))
        inputs = self.to_out_linear(inputs)
        # print(inputs.size())
        # input('oo')
        return inputs

class TempAttNet(torch.nn.Module):
    def __init__(self, num_segment, num_cut, num_rank, att_bottleneck):
        super(TempAttNet, self).__init__()
        self.num_segment = num_segment
        self.num_cut = num_cut
        self.num_rank = num_rank
        self.att_bottleneck = att_bottleneck
        self.att_conv = nn.Sequential(nn.Conv2d(self.num_segment*self.num_cut, 
                    self.att_bottleneck, 
                    kernel_size=2), 
                    nn.ReLU(), 
                    nn.Conv2d(self.att_bottleneck, 
                        self.num_segment*self.num_rank, 
                        kernel_size=2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.activation = nn.ReLU()
        self.activation = nn.Sigmoid()
        # self.activation = nn.Softmax(dim=1)

    def forward(self, inputs):
        '''
        in_feature: (batch x num_seg, in_channel, h, w)
        out_feature: (batch, num_seg, num_rank)
        '''
        inputs = inputs[:, :self.num_cut, ...]
        inputs = inputs.contiguous()
        inputs = inputs.view((-1, self.num_segment*self.num_cut)+\
                inputs.size()[-2:])
        inputs = self.att_conv(inputs)
        # print(inputs.size())
        # input('uu')
        inputs = self.avg_pool(inputs)
        inputs = inputs.view((-1, self.num_segment, self.num_rank))
        inputs = self.activation(inputs)
        return inputs

# class TempPairProposalNet(torch.nn.Module):
    # def __init__(self, num_segment, num_cut):
        # super(TempPairProposalNet, self).__init__()
        # self.num_segment = num_segment
        # self.num_cut = num_cut

    # def forward(self, inputs):
        # '''
        # in_feature: (batch x num_seg, in_channel, h, w)
        # '''
        # inputs = inputs[:, :self.num_cut, ...]
        # inputs = inputs.view((-1, self.num_segment*self.num_cut)+\
                # inputs.size()[-2:])
        # inputs = self.conv_net(inputs)
        # inputs = inputs.view((-1, self.prop_features))
        # print(inputs.size())
        # input('oo')
        # inputs = self.loc_linear(inputs)
        # return inputs


