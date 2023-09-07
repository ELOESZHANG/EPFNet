import torch
from torch import nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation
from u2net_utils import U2NET
from modules import MIE, PIF
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 10, 32, 9 + 3, [32, 32, 64], False)  # origianal radius 0.1 F_1(1024,64)
        self.sa2 = PointNetSetAbstraction(256, 40, 32, 64 + 3, [64, 64, 128], False)  # origianal radius 0.2 F_2(256,128)
        self.sa3 = PointNetSetAbstraction(64, 80, 32, 128 + 3, [128, 128, 256], False)  # origianal radius 0.4  F_3(64,256)
        self.sa4 = PointNetSetAbstraction(16, 100, 32, 256 + 3, [256, 256, 512], False)  # origianal radius 0.8 F_4(16,512)

        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.side3 = U2NET(512,3)
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.side4 = U2NET(256,3)
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.side5 = U2NET(256,3)
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.side6 = U2NET(128,3)

        self.conv1 = nn.Conv1d(129, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

        self.deep_supervision = Upsample(512, 2)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)#F1
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)#F2
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#F3
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)#F4

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

        l3_image = self.side3()
        l2_image = self.side4()
        l1_image = self.side5()
        l0_image = self.side6()

        mie1 = MIE(l3_points, l3_image)
        mie2 = MIE(l3_points, l2_image)
        mie3 = MIE(l3_points, l1_image)
        mie4 = MIE(l3_points, l0_image)

        multi_scale = self.rfb2(l1_xyz, l1_points)

        multi_scale = self.fp1(l0_xyz, l1_xyz, None, multi_scale)

        enhanced_multi_scale = PIF((mie4, multi_scale), dim=1)

        enhanced_multi_scale = self.drop1(F.relu(self.bn1(self.conv1(enhanced_multi_scale))))
        pred = self.conv2(enhanced_multi_scale)
        pred_o = pred.transpose(2, 1).contiguous()
        pred = F.log_softmax(pred_o, dim=-1)
        return pred, pred_o


