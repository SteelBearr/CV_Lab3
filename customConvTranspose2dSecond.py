#Подключение модулей
import numpy as np
import torch

def customConvTranspose2dSecond(in_channels, out_channels, kernel_size, stride=1, padding=0\
                              , output_padding=0, bias=True, dilation=1, padding_mode='zeros'):
    #Обёртка
    def wrapper(mat):
        #Всегда
        internal_stride = 1
        pad_size = kernel_size - 1
        
        temp = []
        for matr in mat:
            zeros = np.zeros((((matr.shape[0] - 1) * (stride) + 1), ((matr.shape[1] - 1) * (stride) + 1)))
            for i in range (0, zeros.shape[0], stride):
                for j in range (0, zeros.shape[1], stride):
                    zeros[i][j] = matr[i // (stride)][j // (stride)]
            pad = np.pad(zeros, pad_width=pad_size, mode='constant')
            temp.append(pad)
        mat = torch.tensor(np.array(temp))

        #Смещение
        if bias:
            bias_value = torch.rand(out_channels)
        else:
            bias_value = torch.zeros(out_channels)

        #Подложка
        if (padding_mode == 'zeros'):
            pad = torch.nn.ZeroPad2d(padding)
            mat = pad(mat)
        elif (padding_mode == 'reflect'):
            pad = torch.nn.ReflectionPad2d(padding)
            mat = pad(mat)
        elif (padding_mode == 'replicate'):
            pad = torch.nn.ReplicationPad2d(padding)
            mat = pad(mat)
        elif (padding_mode == 'circular'):
            pad = torch.nn.CircularPad2d(padding)
            mat = pad(mat)
        else:
            raise Exception(f"Ivalid padding_mode")

        #Фильтр с учётом размера ядра
        if type(kernel_size) == tuple:
            flter = torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1])
        elif type(kernel_size) == int:
            flter = torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        else:
            raise Exception(f"Ivalid kernel_size type")

        #Инверсия фльтра для транспонированной свёртки
        flter_inv = []
        for j in range(out_channels):
            flter_in = []
            for i in range(in_channels):
                flter_in.append(np.flip(np.array(flter[j][i])))
            flter_inv.append(flter_in)
        flter_inv = np.array(flter_inv)
        flter_inv = flter_inv.reshape(in_channels, out_channels, kernel_size, kernel_size)

        #"Обход" фильтром
        res = []
        for chnl in range(out_channels):
            feature_map = np.array([])
            for i in range(0, mat.shape[1] - ((flter.shape[2]- 1) * dilation + 1) + 1, internal_stride):
                for j in range(0, mat.shape[2] - ((flter.shape[3]- 1) * dilation + 1) + 1, internal_stride):
                    total = 0
                    for k in range(in_channels):
                        cur = mat[k]\
                        [i:i + (flter.shape[2] - 1) * dilation + 1 : dilation,\
                         j:j + + (flter.shape[3] - 1) * dilation + 1 : dilation]
                        
                        total += (cur * flter[chnl][k]).sum()
                    feature_map = np.append(feature_map, float(total + bias_value[chnl]))
            res.append(feature_map.reshape((mat.shape[1] - ((flter.shape[2] - 1) * dilation + 1)) // internal_stride + 1,\
                          (mat.shape[2] - ((flter.shape[3] - 1) * dilation + 1)) // internal_stride + 1))

        return np.array(res), np.array(flter_inv), np.array(bias_value)
    return wrapper
