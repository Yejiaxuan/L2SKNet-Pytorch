from torch import nn

import os
from loss import SoftIoULoss
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name, use_morphology=False):
        super(Net, self).__init__()

        self.model_name = model_name
        self.use_morphology = use_morphology
        self.cal_loss = SoftIoULoss()

        if model_name == 'L2SKNet_UNet':
            self.model = L2SKNet_UNet(use_morphology=use_morphology)
        elif model_name == 'L2SKNet_FPN':
            self.model = L2SKNet_FPN(use_morphology=use_morphology)

        elif model_name == 'L2SKNet_1D_UNet':
            self.model = L2SKNet_1D_UNet()  # 1D版本不使用形态学模块
        elif model_name == 'L2SKNet_1D_FPN':
            self.model = L2SKNet_1D_FPN()  # 1D版本不使用形态学模块

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
