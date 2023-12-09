from customConvTranspose2d import customConvTranspose2d
import numpy as np
import torch

test_data_1 = torch.rand(3, 28, 28)
test_data_2 = torch.rand(8, 5, 6)
test_data_3 = torch.rand(1, 1, 1)


def test1():
    customConv = customConvTranspose2d(in_channels=3, out_channels=2, kernel_size=3, stride=10, padding=0\
                                      , output_padding=0, bias=True, dilation=3, padding_mode='zeros')
    
    result, flter, bias_value = customConv(test_data_1)
    torchConv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=2, kernel_size=3, stride=10, padding=0\
                                      , output_padding=0, bias=True, dilation=3, padding_mode='zeros')
    
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_1).data),2))
    assert customResult == torchResult

def test2():
    customConv = customConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=0\
                                      , output_padding=0, bias=True, dilation=1, padding_mode='zeros')
    
    result, flter, bias_value = customConv(test_data_2)
    torchConv = torch.nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=0\
                                      , output_padding=0, bias=True, dilation=1, padding_mode='zeros')
    
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_2).data),2))
    assert customResult == torchResult
    
def test3():
    customConv = customConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=10, padding=0\
                                        , output_padding=0, bias=True, dilation=1, padding_mode='zeros')
    
    result, flter, bias_value = customConv(test_data_3)
    torchConv = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=1, stride=10, padding=0\
                                        , output_padding=0, bias=True, dilation=1, padding_mode='zeros')
    
    torchConv.weight.data = torch.tensor(flter)
    torchConv.bias.data = torch.tensor(bias_value)
    customResult = str(np.round(result,2))
    torchResult = str(np.round(np.array(torchConv(test_data_3).data),2))
    assert customResult == torchResult
    
test1()
test2()
test3()
