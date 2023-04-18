import mindspore.nn as nn
'''import os, sys
sys.path.append(os.getcwd())'''
from models.PS_parts import *

class PSNet(nn.Cell):
    def __init__(self, n_channels=6):
        super(PSNet, self).__init__()

        self.inc = inconv(n_channels, 12)
        self.down1 = down(12, 24)
        self.down2 = down(24, 48)
        self.down3 = down(48, 96)
        self.down4 = down(96, 96)
        self.tas = MLP_tas(64, 2)

    def construct(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.tas(x5)
        return x5, x
    
if __name__ == '__main__':
    import mindspore as ms
    from mindspore.common.initializer import One, Normal

    psnet_model = PSNet(n_channels=9) 
    com_feature_12_u = ms.Tensor(shape=(2, 9, 1024), dtype=ms.float32, init=Normal()) # (2B, 9, 1024)
    u_fea_96, transits_pred = psnet_model(com_feature_12_u)
    print(f'u_fea_96.shape: {u_fea_96.shape}')
    print(f'transits_pred.shape: {transits_pred.shape}')