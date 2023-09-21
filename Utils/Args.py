import argparse

# From: https://stackoverflow.com/a/43357954
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Helper function, such that we can easily share the same argument parser throughout different files.

class GlobalArgumentParser(argparse.ArgumentParser):
    def __init__(self, 
                prog = 'Keypoint Regularized Training for Semantic Segmentation',
                description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                epilog = 'Arguments can be used to overwrite values in a config file.'):
        argparse.ArgumentParser.__init__(self, prog, description, epilog)
                    
        self.add_argument("--scene_path", type=str)

        self.add_argument("--n_depthmaps", type=int, help="The number of depth maps to generate. Should rise with the complexity of the scene")
        self.add_argument("--variational_epsilon", type=float, help="Add this as a constant value to the initial sampling map to also sample from positions that do not vary in depth.")

        self.add_argument("--n_beams", type=int, help="The number of laser beams.")
        self.add_argument("--spp", type=int, help="Define how many samples per pixel should be used in the path tracer.")
        self.add_argument("--sigma", type=float, help="Defines the diameter of the laser beams.")

        self.add_argument("--n_upsamples", type=int, help="How many times the images should be upsampled.")
        self.add_argument("--sequential", type=str2bool, nargs='?', help="Defines, if an animation is loaded sequentially.")
        self.add_argument("--steps_per_anim", type=int, help="How many frames should be generated per animation step.")

        self.add_argument("--iterations", type=int, help="Number of optimization iterations.")
        self.add_argument("--lr_model", type=float, help="Learning rate of the neural network")
        self.add_argument("--lr_laser", type=float, help="Learning rate of the laser rays.")
        self.add_argument("--lr_sigma", type=float, help="Learning rate of the lasers sigma. Set to 0 if the laser spot size should not change.")    
        self.add_argument("--epipolar_constraint_lambda", type=float, help="Defines the weight given to the epipolar constraint regularization. A higher value will disperse the laser beams, a very low value will have basically no effect.")

        self.add_argument("--save_images", type=str2bool, nargs='?', help="Should a video be saved?")
        self.add_argument("--visualize", type=str2bool, nargs='?', help="Should a video be saved?")
        self.add_argument("--upscale", type=float, help="Should a video be saved?")



if __name__ == "__main__":
    parser = GlobalArgumentParser(
                    prog = 'Keypoint Regularized Training for Semantic Segmentation',
                    description = 'Train a Segmentation Network that is optimized for simultaneously outputting keypoints',
                    epilog = 'Arguments can be used to overwrite values in a config file.')
    
    args = parser.parse_args()
    print(args)