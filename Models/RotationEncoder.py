import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import math


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, encoder, out_channels, features):
        super(Decoder, self).__init__()
        self.ups = nn.ModuleList()
        self.encoder = encoder
        self.out_channels = out_channels

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x):
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = self.encoder.skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.in_channels = in_channels

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
            print(x.shape)
            self.skip_connections.append(x)
            x = self.pool(x)

        self.skip_connections = self.skip_connections[::-1]

        return x


class GatedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding_mode="reflect",
        act_fun=nn.ELU,
        normalization=nn.BatchNorm2d,
    ):
        super().__init__()
        self.pad_mode = padding_mode
        self.filter_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        n_pad_pxl = int(self.dilation * (self.filter_size - 1) / 2)

        # this is for backward campatibility with older model checkpoints
        self.block = nn.ModuleDict(
            {
                "conv_f": nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=n_pad_pxl,
                ),
                "act_f": act_fun(),
                "conv_m": nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=n_pad_pxl,
                ),
                "act_m": nn.Sigmoid(),
                "norm": normalization(out_channels),
            }
        )

    def forward(self, x, *args, **kwargs):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)

        return output


class ShapeOutput(nn.Module):
    def __init__(self, input_features, features):
        super(ShapeOutput, self).__init__()


class Model(nn.Module):
    def __init__(self, config, state_dict=None, pretrain=False, device="cuda"):
        super(Model, self).__init__()
        input_features = config["in_channels"]

        features = config["features"]

        self.bottleneck_size = features[-1] * 2

        self.encoder = Encoder(input_features, features)
        self.relu = nn.ReLU()
        self.image_to_rot = nn.Linear(256, 1)

        if state_dict:
            self.load_from_dict(state_dict)

        if pretrain:
            self.encoder.requires_grad_ = False

    def get_statedict(self):
        return {"Encoder": self.encoder.state_dict()}

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(x)
        x = self.image_to_rot(x.squeeze(dim=[-1, -2]))
        return x


class PeakFinder(nn.Module):
    def __init__(self, config, state_dict=None):
        super(PeakFinder, self).__init__()
        input_features = config["in_channels"]
        features = config["features"]
        self.bottleneck_size = features[-1] * 2
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_features * 3, 32))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(32, 16))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(16, 8))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(8, 2))

        if state_dict:
            self.load_from_dict(state_dict)

    def get_statedict(self):
        return {"Layer": self.layers.state_dict()}

    def load_from_dict(self, dict):
        self.layers.load_state_dict(dict["Layer"])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


def test():
    config = {"in_channels": 1, "out_channels": 1, "features": [32, 64, 128]}

    rand_input = torch.randn((4, 1, 256, 256))
    expected_output_size = torch.randn((1, 256, 256))

    model = Model(config)
    bla = model(rand_input)

    print(type(model).__name__)
    print(bla.shape)

    # print(Losses.CountingLoss(points.reshape(4, 2, -1), y))
    # print(CHM_loss.apply(points.reshape(4, 2, -1), y))

    # Seems to be working


if __name__ == "__main__":
    test()
