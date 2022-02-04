from typing import Tuple
import torch
from torch import Tensor, nn

from torch.nn import (
    BatchNorm2d,
    Conv2d,
    PReLU,
    Sequential,
    Dropout,
    Flatten,
    Linear,
    BatchNorm1d,
    MaxPool2d,
    AdaptiveAvgPool2d,
    ReLU,
    Sigmoid,
)

from utils.config import config


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


def get_block(in_channel, depth, num_units, stride=2):
    return [bottleneck_IR_SE(in_channel, depth, stride)] + [
        bottleneck_IR_SE(depth, depth, 1) for i in range(num_units - 1)
    ]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64)
        )
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.6),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512, affine=True),
        )

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


class IDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.facenet = Backbone()
        self.facenet.load_state_dict(torch.load(config.PSP_PRETRAINED / "arcface.pt"))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x: Tensor) -> Tensor:
        x = x[:, :, 35:223, 32:220]  # Crop interesting region?!
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat: Tensor, y: Tensor, x: Tensor) -> Tuple[float, float]:
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        # id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            # diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            # id_logs.append(
            #     {
            #         "diff_target": float(diff_target),
            #         "diff_input": float(diff_input),
            #         "diff_views": float(diff_views),
            #     }
            # )
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count
