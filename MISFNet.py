import torch
import torch.nn as nn


from utils.tensor_ops import cus_sample, upsample_add
from backbone.VGG import (
    Backbone_VGG_in1,
    Backbone_VGG_in3,
)
from module.MyModules import (
    IDEM,
    FDM,
)
import warnings
warnings.filterwarnings("ignore")

class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)

class MISFNet(nn.Module):
    def __init__(self, pretrained=True):
        super(MISFNet, self).__init__()
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.encoder1, self.encoder2, self.encoder4, self.encoder8, self.encoder16 = Backbone_VGG_in3(
            pretrained=pretrained
        )
        (
            self.depth_encoder1,
            self.depth_encoder2,
            self.depth_encoder4,
            self.depth_encoder8,
            self.depth_encoder16,
        ) = Backbone_VGG_in1(pretrained=pretrained)

        self.channel_reduce_1 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.channel_reduce_2= nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.channel_reduce_3 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        # self.channel_reduce_4 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)

        self.trans16 = nn.Conv2d(512, 64, 1)
        self.trans8 = nn.Conv2d(512, 64, 1)
        self.trans4 = nn.Conv2d(256, 64, 1)
        self.trans2 = nn.Conv2d(128, 64, 1)
        self.trans1 = nn.Conv2d(64, 32, 1)


        self.t_trans16 = IDEM(512, 64)
        self.t_trans8 = IDEM(512, 64)
        self.t_trans4 = IDEM(256, 64)
        self.t_trans2 = IDEM(128,32)
        self.t_trans1 = IDEM(64,64)

        self.upconv16 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv8 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = BasicConv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.upconv1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)


        ##########################################
        # shared encoder
        ##########################################
        self.shared_1 = nn.Sequential()
        self.shared_1.add_module('shared_1', nn.Conv2d(64, 64,kernel_size=3,padding=1))
        # self.shared_1.add_module('shared_bn_1', nn.BatchNorm2d(64))
        self.shared_1.add_module('shared_1_activation', nn.ReLU(inplace=True))
        self.shared_2 = nn.Sequential()
        self.shared_2.add_module('shared_2', nn.Conv2d(128, 128,kernel_size=3,padding=1))
        # self.shared_2.add_module('shared_2_bn', nn.BatchNorm2d(128))
        self.shared_2.add_module('shared_2_activation', nn.ReLU(inplace=True))
        self.shared_4 = nn.Sequential()
        self.shared_4.add_module('shared_4', nn.Conv2d(256, 256,kernel_size=3,padding=1))
        # self.shared_4.add_module('shared_bn_1', nn.BatchNorm2d(256))
        self.shared_4.add_module('shared_4_activation', nn.ReLU(inplace=True))
        self.shared_8 = nn.Sequential()
        self.shared_8.add_module('shared_8', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_8.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.shared_8.add_module('shared_8_activation', nn.ReLU(inplace=True))
        self.shared_16 = nn.Sequential()
        self.shared_16.add_module('shared_16', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.shared_16.add_module('shared_16_activation', nn.ReLU(inplace=True))

        ##########################################
        #private encoder
        ##########################################
        self.private_16 = nn.Sequential()
        self.private_16.add_module('private_16', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_16.add_module('private_16_activation', nn.ReLU(inplace=True))

        self.private_16_t = nn.Sequential()
        self.private_16_t.add_module('private_16_t', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_16_t.add_module('private_16_t_activation', nn.ReLU(inplace=True))

        self.private_8 = nn.Sequential()
        self.private_8.add_module('private_8', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_8.add_module('private_8_activation', nn.ReLU(inplace=True))

        self.private_8_t = nn.Sequential()
        self.private_8_t.add_module('private_8_t', nn.Conv2d(512, 512, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_8_t.add_module('private_8_t_activation', nn.ReLU(inplace=True))

        self.private_4 = nn.Sequential()
        self.private_4.add_module('private_4', nn.Conv2d(256, 256, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_4.add_module('private_4_activation', nn.ReLU(inplace=True))

        self.private_4_t = nn.Sequential()
        self.private_4_t.add_module('private_4_t', nn.Conv2d(256, 256, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_4_t.add_module('private_4_t_activation', nn.ReLU(inplace=True))

        self.private_2= nn.Sequential()
        self.private_2.add_module('private_2', nn.Conv2d(128, 128, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_2.add_module('private_2_activation', nn.ReLU(inplace=True))

        self.private_2_t = nn.Sequential()
        self.private_2_t.add_module('private_2_t', nn.Conv2d(128, 128, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_2_t.add_module('private_2_t_activation', nn.ReLU(inplace=True))

        self.private_1 = nn.Sequential()
        self.private_1.add_module('private_1', nn.Conv2d(64, 64, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_1.add_module('private_1_activation', nn.ReLU(inplace=True))

        self.private_1_t = nn.Sequential()
        self.private_1_t.add_module('private_1_t', nn.Conv2d(64, 64, kernel_size=3, padding=1))
        # self.shared_16.add_module('shared_bn_1', nn.BatchNorm2d(512))
        self.private_1_t.add_module('private_1_t_activation', nn.ReLU(inplace=True))


        self.fdm = FDM()



    def forward(self, RGBT):
        in_data = RGBT[0]
        in_depth = RGBT[1]
        in_data_1 = self.encoder1(in_data)


        del in_data
        in_data_1_d = self.depth_encoder1(in_depth)
        del in_depth

        in_data_2 = self.encoder2(in_data_1)
        in_data_2_d = self.depth_encoder2(in_data_1_d)
        in_data_4 = self.encoder4(in_data_2)
        in_data_4_d = self.depth_encoder4(in_data_2_d)


        in_data_8 = self.encoder8(in_data_4)
        in_data_8_d = self.depth_encoder8(in_data_4_d)
        in_data_16 = self.encoder16(in_data_8)
        in_data_16_d = self.depth_encoder16(in_data_8_d)

        in_data_16_shared=self.shared_16(in_data_16)
        in_data_8_shared = self.shared_8(in_data_8)
        in_data_4_shared = self.shared_4(in_data_4)
        in_data_2_shared = self.shared_2(in_data_2)
        in_data_1_shared = self.shared_1(in_data_1)

        in_data_16t_shared=self.shared_16(in_data_16_d)
        in_data_8t_shared = self.shared_8(in_data_8_d)
        in_data_4t_shared = self.shared_4(in_data_4_d)
        in_data_2t_shared = self.shared_2(in_data_2_d)
        in_data_1t_shared = self.shared_1(in_data_1_d)

        in_data_16_private = self.private_16(in_data_16)+in_data_16
        in_data_8_private = self.private_8(in_data_8)+in_data_8
        in_data_4_private = self.private_4(in_data_4)+in_data_4
        in_data_2_private = self.private_2(in_data_2)+in_data_2
        in_data_1_private= self.private_1(in_data_1)+in_data_1

        in_data_16t_private = self.private_16_t (in_data_16_d)
        in_data_8t_private = self.private_8_t (in_data_8_d)
        in_data_4t_private = self.private_4_t (in_data_4_d)
        in_data_2t_private= self.private_2_t (in_data_2_d)
        in_data_1t_private= self.private_1_t (in_data_1_d)


        out_1_s = self.t_trans1(in_data_1_shared,in_data_1t_shared)
        # in_data_16_aux = self.t_trans16(in_data_16, in_data_16_d)
        out_2_s = self.t_trans2(in_data_2_shared,in_data_2t_shared)
        out_4_s = self.t_trans4(in_data_4_shared, in_data_4t_shared)
        out_8_s = self.t_trans8(in_data_8_shared, in_data_8t_shared)
        out_16_s = self.t_trans16(in_data_16_shared, in_data_16t_shared)


        out_1_p=in_data_1t_private+in_data_1_private
        out_2_p = in_data_2t_private + in_data_2_private
        out_4_p = in_data_4t_private + in_data_4_private
        out_8_p = in_data_8t_private + in_data_8_private
        out_16_p = in_data_16t_private + in_data_16_private

        out_1 = out_1_p+out_1_s
        out_2 = out_2_p+out_2_s
        out_4 = out_4_p+out_4_s
        out_8 = out_8_p+out_8_s
        out_16 = out_16_p+out_16_s+in_data_16_d

        out_data = self.fdm(out_1,out_2,out_4,out_8,out_16)

        return out_data, in_data_16_shared,in_data_8_shared,\
               in_data_4_shared,in_data_2_shared,in_data_1_shared,\
               in_data_16t_shared,in_data_8t_shared,in_data_4t_shared,\
               in_data_2t_shared,in_data_1t_shared,\
               in_data_16t_private,in_data_16_private,\
               in_data_8t_private,in_data_8_private,in_data_4t_private,in_data_4_private,in_data_2t_private,in_data_2_private, \
               in_data_1, in_data_1_d, in_data_2, in_data_2_d, in_data_4, in_data_4_d, \
               in_data_8, in_data_8_d, in_data_16, in_data_16_d,\
               in_data_1t_private,in_data_1_private



def fusion_model():
    model = MISFNet()
    return model

if __name__ == "__main__":
    model = MISFNet()
    x = torch.randn(2,3,256,256)
    depth = torch.randn(2,3,256,256)
    fuse = model([x,depth])
    print(fuse.shape)
