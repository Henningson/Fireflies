import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import Objects.flame_pytorch.flame as flame
from argparse import Namespace


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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.in_channels = in_channels

        # Downsampling
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        self.skip_connections = []
        for down in self.downs:
            x = down(x)
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


class Model(nn.Module):
    def __init__(
        self,
        config={
            "num_points": 100,
            "in_channels": 1,
            "out_channels": 1,
            "features": [32, 64, 128],
            "output_image_size": [512, 256],
            "shape_params": 100,
            "expression_params": 100,
        },
        state_dict=None,
        pretrain=False,
        device="cuda",
    ):

        super(Model, self).__init__()
        try:
            in_channels = config["in_channels"]
        except:
            in_channels = 3

        try:
            out_channels = config["out_channels"]
        except:
            out_channels = 4

        features = config["features"]

        flame_config = Namespace(
            **{
                "batch_size": 1,
                "dynamic_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_dynamic_embedding.npy",
                "expression_params": config["expression_params"],
                "flame_model_path": "./Objects/flame_pytorch/model/generic_model.pkl",
                "num_worker": 4,
                "optimize_eyeballpose": True,
                "optimize_neckpose": True,
                "pose_params": 6,
                "ring_loss_weight": 1.0,
                "ring_margin": 0.5,
                "shape_params": config["shape_params"],
                "static_landmark_embedding_path": "./Objects/flame_pytorch/model/flame_static_embedding.pkl",
                "use_3D_translation": True,
                "use_face_contour": True,
            }
        )

        self._flame = flame.FLAME(flame_config)
        self._pose_params = torch.zeros(1, 6, device=device)

        self.output_image_size = config["output_image_size"]
        self.bottleneck_size = features[-1] * 2

        self.in_channels = in_channels
        self.laserbeam_to_2d = nn.Linear(
            config["num_beams"] * 3,
            self.output_image_size[0]
            * self.output_image_size[1]
            * config["in_channels"],
        )
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(self.encoder, out_channels, features)
        self.bottleneck = DoubleConv(features[-1], self.bottleneck_size)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        self.expression_layer = nn.Linear(
            self.output_image_size[0] * self.output_image_size[1] * out_channels,
            config["expression_params"],
        )
        self.shape_layer = nn.Linear(
            self.output_image_size[0] * self.output_image_size[1] * out_channels,
            config["shape_params"],
        )

        if state_dict:
            self.load_from_dict(state_dict)

        if pretrain:
            self.encoder.requires_grad_ = False

    def get_statedict(self):
        return {
            "Encoder": self.encoder.state_dict(),
            "Bottleneck": self.bottleneck.state_dict(),
            "Decoder": self.decoder.state_dict(),
            "LastConv": self.final_conv.state_dict(),
        }

    def load_from_dict(self, dict):
        self.encoder.load_state_dict(dict["Encoder"])
        self.bottleneck.load_state_dict(dict["Bottleneck"])
        self.decoder.load_state_dict(dict["Decoder"])

        try:
            self.final_conv.load_state_dict(dict["LastConv"])
        except:
            print("Final conv not initialized.")

    def forward(self, x):
        x = self.laserbeam_to_2d(x.flatten(start_dim=1)).reshape(
            x.shape[0],
            self.in_channels,
            self.output_image_size[0],
            self.output_image_size[1],
        )
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.final_conv(x)

        shape_estimates = self.shape_layer(x.flatten(start_dim=1))
        expression_estimates = self.expression_layer(x.flatten(start_dim=1))

        vertices, _ = self._flame(
            shape_estimates, expression_estimates, self._pose_params
        )

        # We directly output the vertex positions here
        return vertices, shape_estimates, expression_estimates


def test():
    config = {
        "in_channels": 1,
        "out_channels": 1,
        "num_beams": 50,
        "features": [32, 64, 128],
        "output_image_size": [512, 256],
        "shape_params": 100,
        "expression_params": 100,
    }

    rand_input = torch.randn((1, 50, 3)).cuda()

    model = Model(config).cuda()
    vertices, shape_estimates, expression_estimates = model(rand_input)

    print(type(model).__name__)
    # print(shape_params.shape)
    # print(expression_params.shape)

    # print(Losses.CountingLoss(points.reshape(4, 2, -1), y))
    # print(CHM_loss.apply(points.reshape(4, 2, -1), y))

    # Seems to be working


if __name__ == "__main__":
    test()
