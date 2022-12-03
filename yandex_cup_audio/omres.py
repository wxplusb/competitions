import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from OS_block import OS_block

def layer_parameter_list_input_change(layer_parameter_list, input_channel):
    
    new_layer_parameter_list = []
    for i, i_th_layer_parameter in enumerate(layer_parameter_list):
        if i == 0:
            first_layer_parameter = []
            for cov_parameter in i_th_layer_parameter:
                first_layer_parameter.append((input_channel,cov_parameter[1],cov_parameter[2]))
            new_layer_parameter_list.append(first_layer_parameter)
        else:
            new_layer_parameter_list.append(i_th_layer_parameter)
    return new_layer_parameter_list
        
    

class SampaddingConv1D_BN(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size):
        super(SampaddingConv1D_BN, self).__init__()
        self.padding = nn.ConstantPad1d((int((kernel_size-1)/2), int(kernel_size/2)), 0)
        self.conv1d = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.bn = nn.BatchNorm1d(num_features=out_channels)
        
    def forward(self, X):
        X = self.padding(X)
        X = self.conv1d(X)
        X = self.bn(X)
        return X

    
class Res_OS_layer(nn.Module):
    def __init__(self,layer_parameter_list,out_put_channel_numebr):
        super(Res_OS_layer, self).__init__()  
        self.layer_parameter_list = layer_parameter_list
        self.net = OS_block(layer_parameter_list,False)
        self.res = SampaddingConv1D_BN(layer_parameter_list[0][0][0],out_put_channel_numebr,1)
        
    def forward(self, X):
        temp = self.net(X)
        shot_cut= self.res(X)
        block = F.relu(torch.add(shot_cut,temp))
        return block
        
        

class OS_CNN_res(nn.Module):
    def __init__(self,layer_parameter_list,n_class, n_layers, few_shot = True):
        super(OS_CNN_res, self).__init__()
        self.output_features_size = n_class
        self.few_shot = few_shot
        self.layer_parameter_list = layer_parameter_list
        self.n_layers = n_layers
        
        out_put_channel_numebr = 0
        for final_layer_parameters in layer_parameter_list[-1]:
            out_put_channel_numebr = out_put_channel_numebr+ final_layer_parameters[1] 
        new_layer_parameter_list = layer_parameter_list_input_change(layer_parameter_list, out_put_channel_numebr)
        
        self.net_1 = Res_OS_layer(layer_parameter_list,out_put_channel_numebr)
        
        self.net_list = []
        for i in range(self.n_layers-1):
            temp_layer = Res_OS_layer(new_layer_parameter_list,out_put_channel_numebr)
            self.net_list.append(temp_layer)
            
        self.net = nn.Sequential(*self.net_list)
        
        self.averagepool = nn.AdaptiveAvgPool1d(1)
        self.hidden = nn.Linear(out_put_channel_numebr, n_class)

    def forward(self, X):
        
        temp = self.net_1(X)
        temp = self.net(temp)
        X = self.averagepool(temp)
        X = X.squeeze_(-1)

        if not self.few_shot:
            X = self.hidden(X)
        return X
        
def get_Prime_number_in_a_range(start, end):
    Prime_list = []
    for val in range(start, end + 1): 
        prime_or_not = True
        for n in range(2, val):
            if (val % n) == 0:
                prime_or_not = False
                break
        if prime_or_not:
            Prime_list.append(val)
    return Prime_list


def get_out_channel_number(paramenter_layer, in_channel, prime_list):
    out_channel_expect = int(paramenter_layer/(in_channel*sum(prime_list)))
    return out_channel_expect


def generate_layer_parameter_list(start,end,paramenter_number_of_layer_list, in_channel = 1):
    prime_list = get_Prime_number_in_a_range(start, end)
    if prime_list == []:
        print('start = ',start, 'which is larger than end = ', end)
    input_in_channel = in_channel
    layer_parameter_list = []
    for paramenter_number_of_layer in paramenter_number_of_layer_list:
        out_channel = get_out_channel_number(paramenter_number_of_layer, in_channel, prime_list)
        
        tuples_in_layer= []
        for prime in prime_list:
            tuples_in_layer.append((in_channel,out_channel,prime))
        in_channel =  len(prime_list)*out_channel
        
        layer_parameter_list.append(tuples_in_layer)
    
    tuples_in_layer_last = []
    first_out_channel = len(prime_list)*get_out_channel_number(paramenter_number_of_layer_list[0], input_in_channel, prime_list)
    tuples_in_layer_last.append((in_channel,first_out_channel,start))
    tuples_in_layer_last.append((in_channel,first_out_channel,start+1))
    layer_parameter_list.append(tuples_in_layer_last)
    return layer_parameter_list

layer_parameter_list = generate_layer_parameter_list(1, 15, [8*128*512, 5*128*256 + 2*256*128], in_channel = 512)
        
torch_OS_CNN_res = OS_CNN_res(layer_parameter_list, 256, 3,False)