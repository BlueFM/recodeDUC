import torch
import torch.nn as nn

no_bias = True
bn_momentum = 0.9995
eps = 1e-6

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=not no_bias)

    def forward(self, x):
        return self.conv(x)

class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x)

class Conv_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv_AC, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv_BN, self).__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=bn_momentum)

    def forward(self, x):
        return self.bn(self.conv(x))

class Conv_BN_AC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(Conv_BN_AC, self).__init__()
        self.conv_bn = Conv_BN(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.relu = ReLU()

    def forward(self, x):
        return self.relu(self.conv_bn(x))

class ResidualFactory_o(nn.Module):
    def __init__(self, in_channels, num_1x1_a, num_3x3_b, num_1x1_c, dilation):
        super(ResidualFactory_o, self).__init__()
        self.branch1 = Conv_BN(in_channels, num_1x1_c, kernel_size=1)
        self.branch2a = Conv_BN_AC(in_channels, num_1x1_a, kernel_size=1)
        self.branch2b = Conv_BN_AC(num_1x1_a, num_3x3_b, kernel_size=3, padding=dilation, dilation=dilation)
        self.branch2c = Conv_BN(num_3x3_b, num_1x1_c, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(branch2a)
        branch2c = self.branch2c(branch2b)
        summ = branch1 + branch2c
        return self.relu(summ)

class ResidualFactory_x(nn.Module):
    def __init__(self, in_channels, num_1x1_a, num_3x3_b, num_1x1_c, dilation):
        super(ResidualFactory_x, self).__init__()
        self.branch2a = Conv_BN_AC(in_channels, num_1x1_a, kernel_size=1)
        self.branch2b = Conv_BN_AC(num_1x1_a, num_3x3_b, kernel_size=3, padding=dilation, dilation=dilation)
        self.branch2c = Conv_BN(num_3x3_b, num_1x1_c, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(branch2a)
        branch2c = self.branch2c(branch2b)
        summ = x + branch2c
        return self.relu(summ)

class ResidualFactory_d(nn.Module):
    def __init__(self, in_channels, num_1x1_a, num_3x3_b, num_1x1_c):
        super(ResidualFactory_d, self).__init__()
        self.branch1 = Conv_BN(in_channels, num_1x1_c, kernel_size=1, stride=2)
        self.branch2a = Conv_BN_AC(in_channels, num_1x1_a, kernel_size=1, stride=2)
        self.branch2b = Conv_BN_AC(num_1x1_a, num_3x3_b, kernel_size=3, padding=1)
        self.branch2c = Conv_BN(num_3x3_b, num_1x1_c, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2a = self.branch2a(x)
        branch2b = self.branch2b(branch2a)
        branch2c = self.branch2c(branch2b)
        summ = branch1 + branch2c
        return self.relu(summ)

class ResnetHDC(nn.Module):
    def __init__(self):
        super(ResnetHDC, self).__init__()
        # group 1
        self.res1_1 = Conv_BN_AC(3, 64, kernel_size=3, stride=2, padding=1)
        self.res1_2 = Conv_BN_AC(64, 64, kernel_size=3, padding=1)
        self.res1_3 = Conv_BN_AC(64, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # group 2
        self.res2a = ResidualFactory_o(128, 64, 64, 256, dilation=1)
        self.res2b = ResidualFactory_x(256, 64, 64, 256, dilation=1)
        self.res2c = ResidualFactory_x(256, 64, 64, 256, dilation=1)

        # group 3
        self.res3a = ResidualFactory_d(256, 128, 128, 512)
        self.res3b1 = ResidualFactory_x(512, 128, 128, 512, dilation=1)
        self.res3b2 = ResidualFactory_x(512, 128, 128, 512, dilation=1)
        self.res3b3 = ResidualFactory_x(512, 128, 128, 512, dilation=1)

        # group 4
        self.res4a = ResidualFactory_o(512, 256, 256, 1024, dilation=2)
        self.res4b = nn.ModuleList(
            [ResidualFactory_x(1024, 256, 256, 1024, dilation=i) 
             for i in [2, 5, 9, 1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5, 9, 1, 2, 5]]
             )

        # group 5
        self.res5a = ResidualFactory_o(1024, 512, 512, 2048, dilation=5)
        self.res5b = ResidualFactory_x(2048, 512, 512, 2048, dilation=9)
        self.res5c = ResidualFactory_x(2048, 512, 512, 2048, dilation=17)


    def forward(self, x):
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self.res1_3(x)
        x = self.pool1(x)

        x = self.res2a(x)
        x = self.res2b(x)
        x = self.res2c(x)

        x = self.res3a(x)
        x = self.res3b1(x)
        x = self.res3b2(x)
        x = self.res3b3(x)

        x = self.res4a(x)
        for layer in self.res4b:
            x = layer(x)

        x = self.res5a(x)
        x = self.res5b(x)
        x = self.res5c(x)

        return x


def test_resnet():
    model = ResnetHDC()
    x = torch.randn(1, 3, 224, 224)
    print(x.shape)
    output = model(x)
    
    # Assert the output shape is correct
    assert output.shape == (1, 2048, 28, 28)
    
    # Assert the output values are within a valid range
    assert torch.all(output >= 0) and torch.all(output <= 1)
    
    print("Tests passed!")

if __name__ == '__main__':
    test_resnet()