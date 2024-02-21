import Utils.utils as utils
import argparse
import yaml
from Utils.printer import Printer


class ConfigArgsParser(dict):
    def __init__(self, config, argparser, *arg, **kw):
        super(ConfigArgsParser, self).__init__(*arg, **kw)
        # We assume config to be a dict
        # First copy it
        for key, value in config.items():
            self[key] = value

        # Next, match every key and value in argparser and overwrite it, if it exists
        for key, value in vars(argparser).items():
            if value is None:
                continue

            if key in config:
                self[key] = value
            else:
                Printer.Warning("Key {0} does not exist in config.".format(key))

    def printFormatted(self):
        for key, value in self.items():
            Printer.KV(key, value)

    def printDifferences(self, config):
        for key, value in self.items():
            if config[key] != value:
                Printer.KV(key, value)
            else:
                Printer.KV2(key, value)

    def asNamespace(self) -> argparse.Namespace:
        return argparse.Namespace(**self)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Keypoint Regularized Training for Semantic Segmentation",
        description="Train a Segmentation Network that is optimized for simultaneously outputting keypoints",
        epilog="Arguments can be used to overwrite values in a config file.",
    )
    parser.add_argument("--scene_path", type=str, default="scenes/RotObject")

    parser.add_argument(
        "--n_depthmaps",
        type=int,
        default=150,
        help="The number of depth maps to generate. Should rise with the complexity of the scene",
    )
    parser.add_argument(
        "--variational_epsilon",
        type=float,
        default=0.001,
        help="Add this as a constant value to the initial sampling map to also sample from positions that do not vary in depth.",
    )

    parser.add_argument("--save_images", type=bool, help="Should a video be saved?")

    parser.add_argument(
        "--n_beams", type=int, default=150, help="The number of laser beams."
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=4,
        help="Define how many samples per pixel should be used in the path tracer.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=12.0,
        help="Defines the diameter of the laser beams.",
    )

    parser.add_argument(
        "--n_upsamples",
        type=int,
        default=3,
        help="How many times the images should be upsampled.",
    )
    parser.add_argument(
        "--sequential",
        type=bool,
        help="Defines, if an animation is loaded sequentially.",
    )

    parser.add_argument(
        "--lr_model",
        type=float,
        default=0.0001,
        help="Learning rate of the neural network",
    )
    parser.add_argument(
        "--lr_laser", type=float, default=0.005, help="Learning rate of the laser rays."
    )
    parser.add_argument(
        "--lr_sigma",
        type=float,
        default=0.000001,
        help="Learning rate of the lasers sigma. Set to 0 if the laser spot size should not change.",
    )
    parser.add_argument(
        "--epipolar_constraint_lambda",
        type=float,
        default=0.0001,
        help="Defines the weight given to the epipolar constraint regularization. A higher value will disperse the laser beams, a very low value will have basically no effect.",
    )

    args = parser.parse_args()
    CONFIG_PATH = "config.yml"
    config = utils.load_config(CONFIG_PATH)
    cap = ConfigArgsParser(config, args)
    cap.printDifferences(config)

    with open("blablabla.yml", "w") as outfile:
        yaml.dump(dict(cap), outfile, default_flow_style=False)
