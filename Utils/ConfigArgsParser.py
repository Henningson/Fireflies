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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--logwandb", action="store_true")

    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)

    parser.add_argument("--model", type=str)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--features", type=int, nargs="+")
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--loss_weights", type=float, nargs="+")
    parser.add_argument("--temporal_regularization_at", type=int)
    parser.add_argument("--temporal_lambda", type=float)
    parser.add_argument("--keypoint_regularization_at", type=int)
    parser.add_argument("--nn_threshold", type=float)
    parser.add_argument("--keypoint_lambda", type=float)
    
    args = parser.parse_args()
    CONFIG_PATH = "config.yml"
    config = utils.load_config(CONFIG_PATH)
    cap = ConfigArgsParser(config, args)
    cap.printDifferences(config)


    with open("blablabla.yml", 'w') as outfile:
        yaml.dump(dict(cap), outfile, default_flow_style=False)