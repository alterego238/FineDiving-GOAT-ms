import mindspore as ms
import mindspore.nn as nn


class MLP_score(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(MLP_score, self).__init__()
        self.activation_1 = nn.ReLU()
        self.layer1 = nn.Dense(in_channel, 256)
        self.layer2 = nn.Dense(256, 64)
        self.layer3 = nn.Dense(64, out_channel)

    def construct(self, x):
        x = self.activation_1(self.layer1(x))
        x = self.activation_1(self.layer2(x))
        output = self.layer3(x)
        return output
    
if __name__ == '__main__':
    import mindspore as ms
    from mindspore.common.initializer import One, Normal
    
    regressor_delta = MLP_score(in_channel=64, out_channel=1)
    input4 = ms.Tensor(shape=(1, 15, 64), dtype=ms.float32, init=Normal())
    output = regressor_delta(input4)
    print(f'output.shape: {output.shape}')
