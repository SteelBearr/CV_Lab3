#Подключение модулей
import numpy as np
import torch

def customConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0\
                          , output_padding=0, bias=True, dilation=1, padding_mode='zeros'):
    #Обёртка
    def wrapper(mat):
        
        #Проверки параметров
        if output_padding < 0:
            raise Exception(f"Padiing should be 0 or positive")
        if stride < 0:
            raise Exception(f"Stride should be 0 or positive")
        if (padding_mode != 'zeros'):
            raise Exception(f"Ivalid padding_mode")
            
        #Смещение
        if bias:
            bias_value = torch.rand(out_channels)
        else:
            bias_value = torch.zeros(out_channels)
            
        
        #Фильтр с учётом размера ядра
        if type(kernel_size) == tuple:
            flter = torch.rand(in_channels, out_channels, kernel_size[0], kernel_size[1])
        elif type(kernel_size) == int:
            flter = torch.rand(in_channels, out_channels, kernel_size, kernel_size)
        else:
            raise Exception(f"Ivalid kernel_size type")
            
            
        #"Обход" фильтром
        res = []
        for ochnl in range(out_channels):
            feature_map = torch.zeros((mat.shape[1] - 1) * stride + dilation * (flter.shape[2] - 1) + 1\
                                            , (mat.shape[2] - 1) * stride + dilation * (flter.shape[3] - 1) + 1)
            
            for ichnl in range(in_channels):
                for i in range(0, mat.shape[1]):
                    for j in range(0, mat.shape[2]):
                        cur = mat[ichnl][i][j]
                        val = cur * flter[ichnl][ochnl]
                        zeros = torch.zeros((flter.shape[2] - 1) * dilation + 1, (flter.shape[3] - 1) * dilation + 1)
                        for k in range(0, zeros.shape[0], dilation):
                            for f in range(0, zeros.shape[1], dilation):
                                zeros[k][f] = val[k // dilation][f // dilation]
                        total = np.add((zeros), feature_map[i * stride:i * stride + dilation * (flter.shape[2] - 1) + 1\
                                                            , j * stride:j * stride + dilation * (flter.shape[3] - 1) + 1])
                        
                        feature_map[i * stride:i * stride + dilation * (flter.shape[2] - 1) + 1\
                                    , j * stride:j * stride + dilation * (flter.shape[3] - 1) + 1] = total

            res.append(np.add(feature_map, np.full((feature_map.shape), bias_value[ochnl])))
        
        for l in range(len(res)):
            if output_padding > 0:
                pad = torch.nn.ConstantPad1d((0, output_padding, 0, output_padding), 0)
                res[l] = pad(res[l])
            res[l] = res[l][padding:res[l].shape[0] - padding, padding:res[l].shape[1] - padding]
            
        return np.array(res), np.array(flter), np.array(bias_value)
    return wrapper
