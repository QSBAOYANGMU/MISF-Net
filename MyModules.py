import torch
from torch import nn

from module.BaseBlocks import BasicConv2d
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")



class IDEM(nn.Module):
    def __init__(self, in_C, out_C):
        super(IDEM, self).__init__()
        # down_factor = in_C // out_C
        # self.fuse_down_mul = BasicConv2d(in_C, in_C, 3, 1, 1)
        # self.res_main = DenseLayer(in_C, in_C, down_factor=down_factor)
        # self.fuse_main = BasicConv2d(in_C, out_C, kernel_size=3, stride=1, padding=1)
        # self.fuse_main1 = BasicConv2d(in_C,out_C,kernel_size=1)

        self.ar3 = BasicConv2d(in_C, in_C, kernel_size=3, dilation=1, padding=1)
        self.ar5 = BasicConv2d(in_C, in_C, kernel_size=3, dilation=2, padding=2)
        self.ar7 = BasicConv2d(in_C, in_C, kernel_size=3, dilation=4, padding=4)
        self.ar11 = BasicConv2d(in_C, in_C, kernel_size=3, dilation=8, padding=8)
        self.conv = nn.Conv2d(in_C, in_C, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_C, in_C, kernel_size=1)


    def forward(self, rgb, depth):
        assert rgb.size() == depth.size()
        rgb_3 = self.ar3(rgb) + self.conv1(rgb)
        depth_3 = self.ar3(depth) + self.conv1(depth)
        rgb_5 = self.ar5(rgb) + self.conv1(rgb)
        depth_5 = self.ar5(depth) + self.conv1(depth)
        rgbt_35 = self.conv(rgb_3*depth_3+rgb_3 + rgb_5) + (rgb_3*depth_3+rgb_3 + rgb_5)
        trgb_35 = self.conv(rgb_3*depth_3+depth_3 + depth_5) + (rgb_3*depth_3+depth_3 + depth_5)
        rgb_7 = self.ar7(rgb) + self.conv1(rgb)
        depth_7 = self.ar7(depth) + self.conv1(depth)
        rgbt_357 = self.conv(rgbt_35*trgb_35+rgbt_35 + rgb_7 ) + (rgbt_35*trgb_35+rgbt_35 + rgb_7 )
        tgbt_357 = self.conv(rgbt_35*trgb_35+trgb_35+depth_7) + (rgbt_35*trgb_35+trgb_35+depth_7)
        rgb_11 = self.ar5(rgb) + self.conv1(rgb)
        depth_11 = self.ar5(depth) + self.conv1(depth)
        rgbt_35711 = self.conv(rgbt_357*tgbt_357+rgbt_357+ rgb_11) + (rgbt_357*tgbt_357+rgbt_357+ rgb_11)
        trgb_35711 = self.conv(rgbt_357*tgbt_357+tgbt_357 + depth_11) + (rgbt_357*tgbt_357+tgbt_357 + depth_11)
        # feat=torch.cat([rgbt_35711,trgb_35711],dim=1)
        feat = rgbt_35711 + trgb_35711+rgb_3+depth_3#+rgb+depth##rgbt_35711 + trgb_35711+rgb_3+depth_3
        # feat=rgb+depth
        # feat = depth
        return feat


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




