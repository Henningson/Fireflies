import argparse

# Helper function, such that we can easily share the same argument parser throughout different files.

class GlobalArgumentParser(argparse.ArgumentParser):
    def __init__(self, 
                prog = 'ProgramName',
                description = 'What the program does',
                epilog = 'Text at the bottom of help'):
        argparse.ArgumentParser.__init__(self, prog, description, epilog)

        self.add_argument("--config", type=str, default="config.yml")
        self.add_argument("--logwandb", action="store_true")
        self.add_argument("--pretrain", action="store_true")
        self.add_argument("--checkpoint_name", type=str)

        self.add_argument("--optimizer", type=str)
        self.add_argument("--checkpoint", type=str)
        
        self.add_argument("--model_depth", type=int)
        self.add_argument("--dataset_name", type=str)
        self.add_argument("--dataset_path", type=str)
       
        self.add_argument("--model", type=str)
        self.add_argument("--learning_rate", type=float)
        self.add_argument("--batch_size", type=int)
        self.add_argument("--features", type=int, nargs="+")
        self.add_argument("--kernel3d_size", type=int)
       
        self.add_argument("--num_epochs", type=int)
        self.add_argument("--loss_weights", type=float, nargs="+")
       
        self.add_argument("--temporal_regularization_at", type=int)
        self.add_argument("--temporal_lambda", type=float)
        self.add_argument("--keypoint_regularization_at", type=int)
        self.add_argument("--nn_threshold", type=float)
        self.add_argument("--keypoint_lambda", type=float)

if __name__ == "__main__":
    parser = GlobalArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    
    args = parser.parse_args()
    print(args)