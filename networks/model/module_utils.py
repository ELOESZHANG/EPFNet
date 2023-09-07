import torch
from torch import nn
from torch_points_kernels import knn


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class SharedMLP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            transpose=False,
            padding_mode='zeros',
            bn=False,
            activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x

class MLP_conv1x1(nn.Module):

    def __init__(self, channels, act_func=None, conv_func=nn.Conv2d):
        super(MLP_conv1x1, self).__init__()
        mlp = []
        for i in range(len(channels) - 1):
            mlp.append(conv_func(channels[i], channels[i+1], 1))
            if act_func is not None:
                mlp.append(act_func)
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        out = self.mlp(x)  #Attentive Aggreation
        return out


class MIE(nn.Module):
    def __init__(self, in_channel, out_channel, neighbors):
        super(MIE, self).__init__()
        self.neighbors = neighbors
        self.mlp1 = SharedMLP(in_channel, out_channel // 2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(10, out_channel // 2, bn=True, activation_fn=nn.ReLU())

    def forward(self, points, image, batch_size ):
        upsample = nn.Upsample(scale_factor=2, mode='nearest')
        avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        upsampled_feature1 = upsample(points)
        upsampled_feature2 = upsample(image)
        mlp_output1 = self.mlp1(upsampled_feature1)
        mlp_output2 = self.mlp2(upsampled_feature2)
        pooled_feature1 = avg_pool(mlp_output1.view(batch_size, -1))
        pooled_feature2 = avg_pool(mlp_output2.view(batch_size, -1))
        features = torch.cat(
            pooled_feature1,
            pooled_feature2)
        features = torch.cat(
            [torch.max(features, dim=-1, keepdim=True)[0],
             torch.mean(features, dim=-1, keepdim=True)
             ],
            dim=-3
        )
        return features


class PIF(nn.Module):
        def __init__(self, in_channel, out_channel, neighbors,point_pooling=False):
            super(PIF, self).__init__()
            self.neighbors = neighbors
            self.mlp = SharedMLP(2 * out_channel, out_channel, bn=True, activation_fn=nn.ReLU())
            self.point_pooling = point_pooling

        def forward(self, points, image, knn_output):

            image_features = MLP_conv1x1(image)
            idx, dist = knn_output
            idx, dist = idx[..., :self.neighbors], dist[..., :self.neighbors]
            B, N, K = idx.size()
            extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
            extended_points = points.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
            neighbors = torch.gather(extended_points, 2, extended_idx)

            concat = torch.cat((
                extended_points,
                neighbors,
                extended_points - neighbors,
                dist.unsqueeze(-3),
                image_features
            ))

            features = torch.cat((
                self.mlp(concat),
                extended_points.expand(B, -1, N, K)
            ), dim=-3)

            features = torch.cat(
                [torch.max(features, dim=-1, keepdim=True)[0],
                 torch.mean(features, dim=-1, keepdim=True)
                 ],
                dim=-3
            )
            features = self.mlp(features)
            return features