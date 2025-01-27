import torch
from torch import nn

from module.BaseBlocks import BasicConv2d
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")


class BasicUpsample(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            # nn.Conv2d(32,32,kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)

class BasicUpsample_L(nn.Module):
    def __init__(self,scale_factor):
        super(BasicUpsample_L, self).__init__()

        self.basicupsample = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor,mode='nearest'),
            # nn.Conv2d(128,32,kernel_size=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU()
        )

    def forward(self,x):
        return self.basicupsample(x)


class FDM(nn.Module):
    def __init__(self,):
        super(FDM, self).__init__()
        self.basicconv1 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=1)
        self.basicconv2 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=1)
        self.upsample1 = nn.Sequential(
            # nn.Upsample(scale_factor=1,mode='nearest'),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32,32,1),
            nn.ReLU()
        )
        # self.upsample1_1 = nn.Sequential(
        #     # nn.Upsample(scale_factor=1,mode='nearest'),
        #     nn.Upsample(scale_factor=2, mode='nearest'),
        #     nn.Conv2d(32, 32, 1),
        #     nn.ReLU()
        # )
        self.basicconv3 = BasicConv2d(in_planes=32,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicconv4 = BasicConv2d(in_planes=64,out_planes=32,kernel_size=3,stride=1,padding=1)
        self.basicupsample16 = BasicUpsample(scale_factor=16)
        self.basicupsample_shared = BasicUpsample_L(scale_factor=16)
        self.basicupsample_shared_t = BasicUpsample_L(scale_factor=16)
        self.basicupsample_private = BasicUpsample_L(scale_factor=16)
        self.basicupsample_private_t = BasicUpsample_L(scale_factor=16)
        self.basicupsample8 = BasicUpsample(scale_factor=8)
        self.basicupsample4 = BasicUpsample(scale_factor=4)
        self.basicupsample2 = BasicUpsample(scale_factor=2)
        # self.basicupsample1 = BasicUpsample(scale_factor=2)
        self.basicupsample1 = BasicUpsample(scale_factor=1)

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv12 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv23 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv34 = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv45 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv55 = nn.Sequential(nn.Conv2d(1024, 512, 3, 1, 1), nn.ReLU(inplace=True))
        # self.convsp = nn.Sequential(nn.Conv2d(32, 512, 3, 1, 1), nn.ReLU(inplace=True))



    def forward(self,out_1,out_2,out_4,out_8,out_16):


        f12 = self.pool(out_1)
        f12 = self.conv12(f12)

        f23 = self.conv23(self.pool(torch.cat((f12, out_2), 1)))
        f34 = self.conv34(self.pool(torch.cat((f23, out_4), 1)))
        f45 = self.conv45(self.pool(torch.cat((f34, out_8), 1)))
        f55 = self.conv55(torch.cat((f45, out_16), 1))

        x1 = F.upsample_bilinear(f55, scale_factor=2)
        x2 = F.upsample_bilinear(f45, scale_factor=2)
        x3 = f34
        x = x1 + x2 + x3
        # x = x1 + x2 + x3
        # Out_Data  =out_data_4+out_data_2+out_data_1
        # Out_Data=Out_Data+in_data_16_shared+in_data_16_t_shared+in_data_16_private+in_data_16_t_private
        # shared_private=in_data_16_shared+in_data_16_t_shared+in_data_16_private+in_data_16_t_private
        # Out_Data=torch.cat([Out_Data,shared_private], dim=1)
        out_data = self.reg_layer(x)


        return out_data




