import torch
import math
import torch.nn as nn
import torch.nn.functional as F

MAX_DEPTH = 81
MAX_DEPTH = 81
DEPTH_OFFSET = 0.1 # This is used for ensuring depth prediction gets into positive range

USE_APEX = False  # Enable if you have GPU with Tensor Cores, otherwise doesnt really bring any benefits.
APEX_OPT_LEVEL = "O2"

BATCH_NORM_MOMENTUM = 0.005
ENABLE_BIAS = True
activation_fn = nn.ELU()


class Reduction(nn.Module):
    def __init__(self, scale, input_filters, is_final=False):
        super(Reduction, self).__init__()
        reduction_count = int(math.log(input_filters, 2)) - 2
        self.reductions = torch.nn.Sequential()
        for i in range(reduction_count):
            if i != reduction_count-1:
                self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                    nn.Conv2d(int(input_filters / math.pow(2, i)), int(input_filters / math.pow(2, i + 1)), 1, 1, 0, bias=ENABLE_BIAS),
                    activation_fn))
            else:
                if not is_final:
                    self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                        nn.Conv2d(int(input_filters / math.pow(2, i)), int(input_filters / math.pow(2, i + 1)), 1, 1, 0, bias=ENABLE_BIAS)))
                else:
                    self.reductions.add_module("1x1_reduc_%d_%d" % (scale, i), nn.Sequential(
                        nn.Conv2d(int(input_filters / math.pow(2, i)), 1, 1, 1, 0, bias=ENABLE_BIAS), nn.Sigmoid()))

    def forward(self, ip):
        return self.reductions(ip)


class LPGLayer(nn.Module):
    def __init__(self, scale):
        super(LPGLayer, self).__init__()
        self.scale = scale
        self.u = torch.arange(self.scale).reshape([1, 1, self.scale]).float()
        self.v = torch.arange(int(self.scale)).reshape([1, self.scale, 1]).float()

    def forward(self, plane_eq):
        plane_eq_expanded = torch.repeat_interleave(plane_eq, int(self.scale), 2)
        plane_eq_expanded = torch.repeat_interleave(plane_eq_expanded, int(self.scale), 3)

        n1 = plane_eq_expanded[:, 0, :, :]
        n2 = plane_eq_expanded[:, 1, :, :]
        n3 = plane_eq_expanded[:, 2, :, :]
        n4 = plane_eq_expanded[:, 3, :, :]

        u = self.u.repeat(plane_eq.size(0), plane_eq.size(2) * int(self.scale), plane_eq.size(3)).cuda()
        u = (u - (self.scale - 1) * 0.5) / self.scale

        v = self.v.repeat(plane_eq.size(0), plane_eq.size(2), plane_eq.size(3) * int(self.scale)).cuda()
        v = (v - (self.scale - 1) * 0.5) / self.scale

        d = n4 / (n1 * u + n2 * v + n3)
        d = d.unsqueeze(1)
        return d


class LPGBlock(nn.Module):
    def __init__(self, scale, input_filters=128):
        super(LPGBlock, self).__init__()
        self.scale = scale

        self.reduction = Reduction(scale, input_filters)

        self.conv = nn.Conv2d(4, 3, 1, 1, 0)
        self.LPGLayer = LPGLayer(scale)

    def forward(self, input):
        input = self.reduction(input)

        plane_parameters = torch.zeros_like(input)
        input = self.conv(input)

        theta = input[:, 0, :, :].sigmoid() * 3.1415926535 / 6
        phi = input[:, 1, :, :].sigmoid() * 3.1415926535 * 2
        dist = input[:, 2, :, :].sigmoid() * MAX_DEPTH

        plane_parameters[:, 0, :, :] = torch.sin(theta) * torch.cos(phi)
        plane_parameters[:, 1, :, :] = torch.sin(theta) * torch.sin(phi)
        plane_parameters[:, 2, :, :] = torch.cos(theta)
        plane_parameters[:, 3, :, :] = dist

        plane_parameters[:, 0:3, :, :] = F.normalize(plane_parameters.clone()[:, 0:3, :, :], 2, 1)

        depth = self.LPGLayer(plane_parameters.float())
        return depth