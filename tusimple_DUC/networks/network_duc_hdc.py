import torch
import torch.nn as nn
from resnet import ResnetHDC

class DUC_HDC(nn.Module):
    def __init__(self, label_num=19, ignore_label=255, aspp_num=4, aspp_stride=6, cell_cap=64, exp="cityscapes"):
        super(DUC_HDC, self).__init__()
        # Base Network
        self.res = ResnetHDC()
        self.label_num = label_num
        
        # ASPP
        self.aspp_list = nn.ModuleList([nn.Conv2d(2048, cell_cap * label_num, kernel_size=3, padding=(i + 1) * aspp_stride, dilation=(i + 1) * aspp_stride) for i in range(aspp_num)])

    def forward(self, x):
        x = self.res(x)

        aspp_out = [aspp(x) for aspp in self.aspp_list]
        summ = sum(aspp_out)

        cls_score_reshape = summ.view(summ.size(0), self.label_num, -1)
        cls = nn.functional.softmax(cls_score_reshape, dim=1)

        return cls

if __name__ == '__main__':
    module = DUC_HDC()

    input = torch.randn(3,3,480,480)
    output = module(input)
    print(output.size())
    